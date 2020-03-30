/*
Copyright (c) 2020 Simon Donné, Max Planck Institute for Intelligent Systems, Tuebingen, Germany

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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