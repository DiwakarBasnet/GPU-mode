#include <torch/extension.h>

void cudaLayerNormalization(
    float *input, 
    float *output, 
    int batch_size, 
    int seq_len, 
    int embed_dim
);
torch::Tensor layerNorm(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int embed_dim = input.size(2);

    torch::Tensor output = torch::zeros_like(input);
    
    cudaLayerNorm(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        embed_dim
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layerNorm", &layerNorm, "Layer Normalization (CUDA)");
}