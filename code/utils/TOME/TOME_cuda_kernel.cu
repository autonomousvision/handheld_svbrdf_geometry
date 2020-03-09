#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "PermutohedralLatticeGPU.cuh"
#include "DeviceMemoryAllocator.h"

#include <vector>

// for kernels that are actually only implemented in single-precision
// (here because of needing atomicMinf)
#define AT_DISPATCH_SINGLE_FLOAT(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const at::Type& the_type = TYPE;                                         \
    switch (the_type.scalarType()) {                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'"); \
    }                                                                        \
  }()


template <typename scalar_t>
__inline__ __device__ scalar_t TOME_get_point_depth(scalar_t* __restrict__ camera, scalar_t* __restrict__ win) {
  return camera[8]*win[0] + camera[9]*win[1] + camera[10]*win[2]+ camera[11];
}

template <typename scalar_t>
__inline__ __device__ bool TOME_project_point(scalar_t* __restrict__ camera, scalar_t* __restrict__ win, int *out, int input_width, int input_height) {
  scalar_t cx = camera[0]*win[0] + camera[1]*win[1] + camera[2]*win[2] + camera[3];
  scalar_t cy = camera[4]*win[0] + camera[5]*win[1] + camera[6]*win[2] + camera[7];
  scalar_t cz = TOME_get_point_depth(camera, win);
  out[0] = int(cx / cz + 0.5f);
  out[1] = int(cy / cz + 0.5f);
  return (out[0] >= 0) && (out[1] >= 0) && (out[0]<input_width) && (out[1]<input_height);
}

template <typename scalar_t>
__inline__ __device__ bool TOME_project_pointf(scalar_t* __restrict__ camera, scalar_t* __restrict__ win, scalar_t* __restrict__ out, int input_width, int input_height) {
  scalar_t cx = camera[0]*win[0] + camera[1]*win[1] + camera[2]*win[2] + camera[3];
  scalar_t cy = camera[4]*win[0] + camera[5]*win[1] + camera[6]*win[2] + camera[7];
  scalar_t cz = TOME_get_point_depth(camera, win);
  out[0] = cx / cz;
  out[1] = cy / cz;
  return (out[0] >= 0) && (out[1] >= 0) && (out[0]<=input_width-1.0f) && (out[1]<=input_height-1.0f);
}

template <typename scalar_t>
__inline__ __device__ void TOME_unproject_point(scalar_t* __restrict__ camloc, scalar_t* __restrict__ invKR, int u, int v, scalar_t z, scalar_t* __restrict__ out) {
  out[0] = camloc[0] + (invKR[0] * u + invKR[1] * v + invKR[2]) * z;
  out[1] = camloc[1] + (invKR[3] * u + invKR[4] * v + invKR[5]) * z;
  out[2] = camloc[2] + (invKR[6] * u + invKR[7] * v + invKR[8]) * z;
}

__device__ static float TOME_atomicMinf(float* addr, float val)
{
  float old;
  old = (val >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(val))) :
       __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(val)));

  return old;
}

// input depth: BxHxW depth tensor
// output depth: BxKxHxW depth tensor
// cameras: BxKx3x4 tensor (receiving cameras)
// invKRs: Bx3x3 tensor (central camera)
// camlocs: Bx3x1 tensor (central camera)
template <typename scalar_t>
__global__ void depth_reprojection_cuda_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ cameras,
    scalar_t* __restrict__ invKRs,
    scalar_t* __restrict__ camlocs,
    int B,
    int K,
    int inH, int inW,
    int outH, int outW)
{
    int proj[2];
    scalar_t wloc[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    scalar_t* camloc = camlocs + b * 3;
    scalar_t* invKR = invKRs + b * 9;
    for (int h = blockIdx.y * blockDim.y + threadIdx.y; h < inH; h += blockDim.y * gridDim.y) {
    for (int w = blockIdx.z * blockDim.z + threadIdx.z; w < inW; w += blockDim.z * gridDim.z) {
        // cast this point into space
        scalar_t depth = input[b * inH * inW + h * inW + w];
        if(depth > 0) {

            for (int k = 0; k < K; k++) {
                scalar_t* camera = cameras + b * K * 12 + k * 12;
                TOME_unproject_point(camloc, invKR, w, h, depth, wloc);
                // project it onto the first camera again
                if(TOME_project_point(camera, wloc, proj, outW, outH)) {
                    TOME_atomicMinf(
                        output + b * K * outH * outW + k * outH * outW + proj[1] * outW + proj[0],
                        TOME_get_point_depth(camera, wloc)
                    );
                }
            }
        }
    }
    }
    }
}

at::Tensor depth_reprojection_cuda(
    at::Tensor input_depth,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    int outH,
    int outW) {

  auto blkdim = 16;
  const auto B = cameras.size(0);
  const auto K = cameras.size(1);
  const auto inH = input_depth.size(1);
  const auto inW = input_depth.size(2);

  const dim3 block = dim3(1, blkdim, blkdim);
  const dim3 grid = dim3(1, 8, 8);
  
  auto output_depth = at::zeros({B, K, outH, outW}, input_depth.type());
  auto sentinel = 1e9;
  output_depth.fill_(sentinel);

  AT_DISPATCH_SINGLE_FLOAT(input_depth.type(), "depth_reprojection_cuda", ([&] {
    depth_reprojection_cuda_kernel<scalar_t><<<grid, block>>>(
        input_depth.data<scalar_t>(),
        output_depth.data<scalar_t>(),
        cameras.data<scalar_t>(),
        invKR.data<scalar_t>(),
        camloc.data<scalar_t>(),
        B, K, inH, inW, outH, outW);
  }));
  
  output_depth.fmod_(sentinel);

  return output_depth;
}

// input depth: BxinHxinW depth tensor
// output depth: BxKxoutHxoutW depth tensor
// cameras: Bx3x4 tensor (central camera)
// invKRs: BxKx3x3 tensor (receiving cameras)
// camlocs: BxKx3x1 tensor (receiving cameras)
template <typename scalar_t>
__global__ void depth_reprojection_bound_cuda_kernel(
    scalar_t *input,
    scalar_t *output,
    scalar_t *cameras,
    scalar_t *invKRs,
    scalar_t *camlocs,
    int B,
    int K,
    int inH,
    int inW,
    int outH,
    int outW,
    scalar_t dmin,
    scalar_t dmax,
    scalar_t dstep)
{
    int proj[2];
    scalar_t wloc[3];

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    for (int k = 0; k < K; k++) {
        scalar_t* camloc = camlocs + b * K * 3 + k * 3;
        scalar_t* invKR = invKRs + b * K * 9 + k * 9;
        scalar_t *camera = cameras + b * 12;

        for (int h = blockIdx.y * blockDim.y + threadIdx.y; h < outH; h += blockDim.y * gridDim.y) {
        for (int w = blockIdx.z * blockDim.z + threadIdx.z; w < outW; w += blockDim.z * gridDim.z) {
            // cast this point into space at increasingly large depths (from camera 0)
            // the first depth at which it is invisible in view n (i.e. lies behind its depth map)
            // that is the lowest permissible depth for this pixel according to that view
            // for very sharp depth edges, this results in an interpolation of the depth map
            // for aliased reprojections, this results in a filling of the holes
            // bool projected_in = false;
            scalar_t dhyp = dmin;
            for (; dhyp <= dmax; dhyp += dstep) {
                TOME_unproject_point(camloc, invKR, w, h, dhyp, wloc);
                // project it onto the first camera again
                if(TOME_project_point(camera, wloc, proj, inW, inH)) {
                    // projected_in = true;
                    scalar_t dhyp_depth_n = TOME_get_point_depth(camera, wloc);
                    scalar_t depth_n = input[b * inH * inW + proj[1] * inW + proj[0]];
                    if(dhyp_depth_n > depth_n && depth_n > 0) {
                        break;
                    }
                }
                // else if (projected_in) {
                //     // just give up -- no value here is acceptable
                //     // dhyp = dmax;
                //     break;
                // }
            }
            if(dhyp < dmax) {
                // refine the estimate
                scalar_t ndhyp = dhyp;
                for (; ndhyp >= dhyp - dstep; ndhyp -= dstep/10) {
                    TOME_unproject_point(camloc, invKR, w, h, ndhyp, wloc);
                    // project it onto the first camera again
                    if(TOME_project_point(camera, wloc, proj, inW, inH)) {
                        scalar_t dhyp_depth_n = TOME_get_point_depth(camera, wloc);
                        scalar_t depth_n = input[b * inH * inW + proj[1] * inW + proj[0]];
                        if(dhyp_depth_n < depth_n) {
                            break;
                        }
                    }
                    else {
                        break;
                    }
                }
                dhyp = ndhyp;
                for (; ndhyp < dhyp + dstep/10; ndhyp += dstep/50) {
                    TOME_unproject_point(camloc, invKR, w, h, ndhyp, wloc);
                    // project it onto the first camera again
                    if(TOME_project_point(camera, wloc, proj, inW, inH)) {
                        scalar_t dhyp_depth_n = TOME_get_point_depth(camera, wloc);
                        scalar_t depth_n = input[b * inH * inW + proj[1] * inW + proj[0]];
                        if(dhyp_depth_n > depth_n && depth_n > 0) {
                            break;
                        }
                    }
                    else {
                        break;
                    }
                }
                dhyp = ndhyp;
            }
            else {
                dhyp = 0.0f;
            }
            output[b * K * outH * outW + k * outH * outW + h * outW + w] = dhyp;
        }
        }
    }
    }
}

at::Tensor depth_reprojection_bound_cuda(
    at::Tensor input_depth,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    int outH, int outW,
    float dmin,
    float dmax,
    float dstep) {

  auto blkdim = 16;
  const auto B = invKR.size(0);
  const auto K = invKR.size(1);
  const auto inH = input_depth.size(1);
  const auto inW = input_depth.size(2);
  
  const dim3 block = dim3(1, blkdim, blkdim);
  const dim3 grid = dim3(1, 8, 8);
  
  auto output_depth = at::zeros({B, K, outH, outW}, input_depth.type());
  auto sentinel = 1e9;
  output_depth.fill_(sentinel);

  AT_DISPATCH_SINGLE_FLOAT(input_depth.type(), "depth_reprojection_bound_cuda_kernel", ([&] {
    depth_reprojection_bound_cuda_kernel<scalar_t><<<grid, block>>>(
          input_depth.data<scalar_t>(),
          output_depth.data<scalar_t>(),
          cameras.data<scalar_t>(),
          invKR.data<scalar_t>(),
          camloc.data<scalar_t>(),
          B, K, inH, inW, outH, outW, dmin, dmax, dstep);
  }));
  
  output_depth.fmod_(sentinel);
  
  return output_depth;
}


// input depth: BxHxW depth tensor
// output depth: BxKxHxW depth tensor
// cameras: BxKx3x4 tensor (receiving cameras)
// invKRs: Bx3x3 tensor (central camera)
// camlocs: Bx3x1 tensor (central camera)
template <typename scalar_t>
__global__ void depth_reprojection_splat_cuda_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ output_depth,
    scalar_t* __restrict__ output_weights,
    scalar_t* __restrict__ cameras,
    scalar_t* __restrict__ invKRs,
    scalar_t* __restrict__ camlocs,
    scalar_t radius,
    scalar_t depth_scale,
    int B,
    int K,
    int inH, int inW,
    int outH, int outW)
{
    scalar_t proj[2];
    scalar_t wloc[3];

    // twice the stddev: 95% of the mass
    int iradius = int(ceil(2*radius));
    scalar_t expdiv = radius>0?2*radius*radius:1.0;

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    scalar_t* camloc = camlocs + b * 3;
    scalar_t* invKR = invKRs + b * 9;
    for (int h = blockIdx.y * blockDim.y + threadIdx.y; h < inH; h += blockDim.y * gridDim.y) {
    for (int w = blockIdx.z * blockDim.z + threadIdx.z; w < inW; w += blockDim.z * gridDim.z) {
        // cast this point into space
        scalar_t depth = input[b * inH * inW + h * inW + w];
        if(depth > 0) {
            TOME_unproject_point(camloc, invKR, w, h, depth, wloc);
            for (int k = 0; k < K; k++) {
                scalar_t* camera = cameras + b * K * 12 + k * 12;
                TOME_project_pointf(camera, wloc, proj, outW, outH);
                scalar_t depth_k = TOME_get_point_depth(camera, wloc);
                int px = int(floor(proj[0]+0.5f));
                int py = int(floor(proj[1]+0.5f));
                for(int xk = max(0, px - iradius); xk <= min(px + iradius, outW-1); xk++) {
                for(int yk = max(0, py - iradius); yk <= min(py + iradius, outH-1); yk++) {
                    scalar_t dist_k = (xk-proj[0])*(xk-proj[0]) + (yk-proj[1])*(yk-proj[1]);
                    // mass: what fraction of the blob in this pixel
                    scalar_t mass_k = exp(-dist_k / expdiv);
                    // weight: softmaxing depth in this pixel
                    scalar_t weight_k = exp(-depth_k / depth_scale);
                    atomicAdd(
                        output_depth + b * K * outH * outW + k * outH * outW + yk * outW + xk,
                        depth_k * mass_k * weight_k
                    );
                    atomicAdd(
                        output_weights + b * K * outH * outW + k * outH * outW + yk * outW + xk,
                        mass_k * weight_k
                    );
                }
                }
            }
        }
    }
    }
    }
}
template <typename scalar_t>
__global__ void depth_reprojection_splat_visibilities_cuda_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ output_depth,
    scalar_t* __restrict__ output_visibilities,
    scalar_t* __restrict__ cameras,
    scalar_t* __restrict__ invKRs,
    scalar_t* __restrict__ camlocs,
    scalar_t radius,
    scalar_t depth_scale,
    int B,
    int K,
    int inH, int inW,
    int outH, int outW)
{
    scalar_t proj[2];
    scalar_t wloc[3];

    // twice the stddev: 95% of the mass
    int iradius = int(ceil(2*radius));
    scalar_t expdiv = radius>0?2*radius*radius:1.0;

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < B; b += blockDim.x * gridDim.x) {
    scalar_t* camloc = camlocs + b * 3;
    scalar_t* invKR = invKRs + b * 9;
    for (int h = blockIdx.y * blockDim.y + threadIdx.y; h < inH; h += blockDim.y * gridDim.y) {
    for (int w = blockIdx.z * blockDim.z + threadIdx.z; w < inW; w += blockDim.z * gridDim.z) {
        // cast this point into space
        scalar_t depth = input[b * inH * inW + h * inW + w];
        if(depth > 0) {
            TOME_unproject_point(camloc, invKR, w, h, depth, wloc);
            for (int k = 0; k < K; k++) {
                scalar_t* camera = cameras + b * K * 12 + k * 12;
                TOME_project_pointf(camera, wloc, proj, outW, outH);
                scalar_t depth_k = TOME_get_point_depth(camera, wloc);
                scalar_t visiblemass_sum = 0;
                scalar_t mass_sum = 0;

                int px = int(floor(proj[0]+0.5f));
                int py = int(floor(proj[1]+0.5f));
                for(int xk = max(0, px - iradius); xk <= min(px + iradius, outW-1); xk++) {
                for(int yk = max(0, py - iradius); yk <= min(py + iradius, outH-1); yk++) {
                    scalar_t dist_k = (xk-proj[0])*(xk-proj[0]) + (yk-proj[1])*(yk-proj[1]);
                    // mass: what fraction of the blob in this pixel
                    scalar_t mass_k = exp(-dist_k / expdiv);
                    scalar_t zbuffer_k = output_depth[b * K * outH * outW + k * outH * outW + yk * outW + xk];
                    
                    // weight: softmaxing depth in this pixel
                    scalar_t visibility_k = exp((zbuffer_k - depth_k) / depth_scale);
                    visibility_k = min(visibility_k, 1.0);

                    visiblemass_sum += mass_k * visibility_k;
                    mass_sum += mass_k;
                }
                }
                if(mass_sum > 0) {
                    output_visibilities[
                        b * K * inH * inW + k * inH * inW + h * inW + w
                    ] = visiblemass_sum / mass_sum;
                }
            }
        }
    }
    }
    }
}

std::vector<at::Tensor> depth_reprojection_splat_cuda(
    at::Tensor input_depth,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    float radius,
    float zbuffer_scale,
    float visibility_scale,
    int outH,
    int outW) {

  auto blkdim = 16;
  const auto B = cameras.size(0);
  const auto K = cameras.size(1);
  const auto inH = input_depth.size(1);
  const auto inW = input_depth.size(2);

  const dim3 block = dim3(1, blkdim, blkdim);
  const dim3 grid = dim3(1, 8, 8);
  
  auto output_depth = at::zeros({B, K, outH, outW}, input_depth.type());
  auto output_weights = at::zeros({B, K, outH, outW}, input_depth.type());
  auto output_visibilities = at::zeros({B, K, inH, inW}, input_depth.type());

  AT_DISPATCH_SINGLE_FLOAT(input_depth.type(), "depth_reprojection_splat_cuda", ([&] {
    depth_reprojection_splat_cuda_kernel<scalar_t><<<grid, block>>>(
        input_depth.data<scalar_t>(),
        output_depth.data<scalar_t>(),
        output_weights.data<scalar_t>(),
        cameras.data<scalar_t>(),
        invKR.data<scalar_t>(),
        camloc.data<scalar_t>(),
        radius, zbuffer_scale,
        B, K, inH, inW, outH, outW);
    output_depth.div_(output_weights);
    depth_reprojection_splat_visibilities_cuda_kernel<scalar_t><<<grid, block>>>(
        input_depth.data<scalar_t>(),
        output_depth.data<scalar_t>(),
        output_visibilities.data<scalar_t>(),
        cameras.data<scalar_t>(),
        invKR.data<scalar_t>(),
        camloc.data<scalar_t>(),
        radius, visibility_scale,
        B, K, inH, inW, outH, outW);
  }));
  

  return {output_depth, output_weights, output_visibilities};
}

at::Tensor permutohedral_filter_cuda(
    at::Tensor input,
    at::Tensor positions,
    at::Tensor weights,
    bool reverse
) {
    auto blkdim = 16;
    const auto H = input.size(0);
    const auto W = input.size(1);
    const auto num_pixels = H*W;
    const dim3 block = dim3(1, blkdim, blkdim);
    const dim3 grid = dim3(1, 8, 8);
    
    const auto pd = positions.size(2);
    const auto id = input.size(2);

    auto output = at::zeros({H, W, id}, input.type());
  
    auto allocator = DeviceMemoryAllocator();

    AT_DISPATCH_SINGLE_FLOAT(input.type(), "permutohedral_filter_cuda", ([&] {
        if(pd == 5 && id == 3) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 5, 4>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 2 && id == 3) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 2, 4>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 2 && id == 2) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 2, 3>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 2 && id == 1) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 2, 2>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 3 && id == 1) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 3, 2>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 3 && id == 2) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 3, 3>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 3 && id == 3) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 3, 4>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 6 && id == 2) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 6, 3>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 6 && id == 3) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 6, 4>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 6 && id == 4) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 6, 5>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 6 && id == 5) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 6, 6>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 6 && id == 6) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 6, 7>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 6 && id == 7) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 6, 8>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else if(pd == 6 && id == 8) {
            auto lattice = PermutohedralLatticeGPU<scalar_t, 6, 9>(num_pixels, &allocator);
            lattice.filter(
                output.data<scalar_t>(),
                input.data<scalar_t>(),
                positions.data<scalar_t>(),
                weights.data<scalar_t>(),
                reverse
            );
        }
        else{
            AT_ASSERTM(false, "permutohedral filter: this (pd,id) is not present in the compiled binary");
        }
    }));
  
    return output;
}