#include <cuda_runtime.h>
#include <math.h>

#define gamma 1.0f
#define beta 0.0f
#define epsilon 1e-6

__global__ void layer_normalization_kernel(
    float *A, float *B, int batch_size, int seq_len, int embed_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int embed_idx = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_var;

    if (threadIdx.x == 0) {

        // Compute mean across the embedding dimension
        float sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            int idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + i;
            sum += A[idx];
        }
        s_mean = sum / embed_dim;

        // Copmute variance across the embedding dimension
        sum = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            int idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + i;
            float diff = A[idx] - s_mean;
            sum += diff * diff;
        }
        s_var = sum / embed_dim;
    }
    
    __syncthreads();

    if (embed_idx < embed_dim) {
        // Normalization
        int idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + embed_idx;
        float normalized = (A[idx] - s_mean) / sqrtf(s_var + epsilon);
        B[idx] = gamma * normalized + beta;
    }
}

void cudaLayerNorm(
    float *input, float *output, int batch_size, int seq_len, int embed_dim
) {
    const dim3 blockDim(embed_dim, 1, 1);
    const dim3 gridDim(batch_size, seq_len, 1);

    layer_normalization_kernel<<<gridDim, blockDim>>>(input, output, batch_size, seq_len, embed_dim);
    cudaDeviceSynchronize();
}