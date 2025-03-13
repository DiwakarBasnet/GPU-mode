#include <torch/extension.h>

void cudaLayerNorm(
    float *input, 
    float *output, 
    int batch_size, 
    int seq_len, 
    int embed_dim,
    float eps
);
torch::Tensor layerNorm(torch::Tensor input, float eps) {
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int embed_dim = input.size(2);

    torch::Tensor output = torch::zeros_like(input);
    
    cudaLayerNorm(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        embed_dim,
        eps
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layerNorm", &layerNorm, "Layer Normalization (CUDA)");
}