#include <torch/extension.h>

void cudaLayerNormalization(float *input, float *output, float *mean, float *variance, int width, int height);
torch::Tensor layerNorm(torch::Tensor input) {
    torch::Tensor output = torch::zeros_like(input);
    const int height = input.size(0);
    const int width = input.size(1);
    cudaLayerNorm(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layerNorm", &layerNorm, "Layer Normalization (CUDA)");
}