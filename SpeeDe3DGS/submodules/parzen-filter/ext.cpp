#include <torch/extension.h>

torch::Tensor parzen_density_cuda(
    torch::Tensor fd_data,
    torch::Tensor ts_data,
    torch::Tensor x_grid,
    torch::Tensor t_grid,
    float h_x,
    float h_t,
    float wk,
    float wlam);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parzen_density",
          &parzen_density_cuda,
          "2D joint Parzen density over (depth x time) grid — returns [N, D, T] (CUDA)");
}
