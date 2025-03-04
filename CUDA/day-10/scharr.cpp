#include <torch/extension.h>

// Function declaration for CUDA kernel
torch::Tensor scharr_cuda_forward(torch::Tensor input);

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // binding scharr cuda forward to the python forward function
    m.def("forward", &scharr_cuda_forward, "Scharr Filter forward pass (CUDA)");
}