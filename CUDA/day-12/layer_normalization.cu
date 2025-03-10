#include <cuda_runtime.h>
#include <math.h>

#define gamma 1.0f
#define beta 0.0f

__global__ void layer_normalization_kernel(
    float *A, float *B, float *mean, float *variance, int width, int height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height) {

        // Compute mean
        float sum = 0.0f;
        for (int col =0; col < width; col++) {
            sum += A[row * width + col];
        }
        mean[row] = sum / width;

        // Copmute variance
        sum = 0.0f;
        for (int col = 0; col < width; col++) {
            sum += (A[row * width + col] - mean[row]) * (A[row * width + col] - mean[row]);
        }
        variance[row] = sum / width;

        // Normalize
        stddev = sqrtf(variance[row] + 1e-6);
        for (int col = 0; col < width; col++) {
            B[row * width + col] = gamma * ((A[row * width + col] - mean[row]) / stddev) + beta;
        }
    }
}

void cudaLayerNorm(
    float *input, float *output, float *mean, float *variance, int width, int height
) {
    const dim3 blockSize(256);
    const dim3 gridSize((height + blockSize.x - 1) / blockSize.x);

    layer_normalization_kernel<<<gridSize, blockSize>>>(input, output, mean, variance, width, height);
    cudaDeviceSynchronize();
}