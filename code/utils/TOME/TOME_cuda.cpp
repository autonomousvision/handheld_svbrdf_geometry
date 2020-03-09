#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

at::Tensor depth_reprojection_cuda(
    at::Tensor input_depth,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    int outH, int outW);

at::Tensor depth_reprojection_bound_cuda(
    at::Tensor input,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    int outH, int outW,
    float dmin,
    float dmax,
    float dstep);

std::vector<at::Tensor> depth_reprojection_splat_cuda(
    at::Tensor input,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    float radius, float zbuffer_scale, float visibility_scale,
    int outH, int outW);

at::Tensor permutohedral_filter_cuda(
    at::Tensor input,
    at::Tensor positions,
    at::Tensor weights,
    bool reverse);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor depth_reprojection(
    at::Tensor input_depth,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    int outH, 
    int outW) {
  CHECK_INPUT(input_depth);
  CHECK_INPUT(cameras);
  CHECK_INPUT(invKR);
  CHECK_INPUT(camloc);

  return depth_reprojection_cuda(input_depth, cameras, invKR, camloc, outH, outW);
}

at::Tensor depth_reprojection_bound(
    at::Tensor input_depth,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    int outH, 
    int outW,
    float dmin,
    float dmax,
    float dstep) {
  CHECK_INPUT(input_depth);
  CHECK_INPUT(cameras);
  CHECK_INPUT(invKR);
  CHECK_INPUT(camloc);

  return depth_reprojection_bound_cuda(input_depth, cameras, invKR, camloc, outH, outW, dmin, dmax, dstep);
}

std::vector<at::Tensor> depth_reprojection_splat(
    at::Tensor input_depth,
    at::Tensor cameras,
    at::Tensor invKR,
    at::Tensor camloc,
    float radius,
    float zbuffer_scale,
    float visibility_scale,
    int outH, 
    int outW
  ) {
  CHECK_INPUT(input_depth);
  CHECK_INPUT(cameras);
  CHECK_INPUT(invKR);
  CHECK_INPUT(camloc);

  return depth_reprojection_splat_cuda(input_depth, cameras, invKR, camloc, radius, zbuffer_scale, visibility_scale, outH, outW);
}

at::Tensor permutohedral_filter(
   at::Tensor input,
   at::Tensor positions,
   at::Tensor weights,
   bool reverse
) {
  CHECK_INPUT(input);
  CHECK_INPUT(positions);

  return permutohedral_filter_cuda(input, positions, weights, reverse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depth_reprojection", &depth_reprojection, "Depth reprojection");
  m.def("depth_reprojection_bound", &depth_reprojection_bound, "Depth reprojection bound");
  m.def("depth_reprojection_splat", &depth_reprojection_splat, "Depth reprojection splat");
  m.def("permutohedral_filter", &permutohedral_filter, "Permutohedral filter");
}