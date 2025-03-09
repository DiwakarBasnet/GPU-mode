#include <torch/extension.h>

void cudaSoftmax(float *input, float *output, int height, int width);
torch::Tensor softmax(torch::Tensor input) {
    torch::Tensor output = torch::zeros_like(input);
    const int height = input.size(0);
    const int width = input.size(1);
    cudaSoftmax(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax, "Softmax (CUDA)");
}