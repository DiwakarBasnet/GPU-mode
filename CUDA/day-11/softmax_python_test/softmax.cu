#include <cuda_runtime.h>

__global__ void softmax_kernel(float *A, float *B, int height, int width){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height) {
        float max_val = -INFINITY;
        float sum_val = 0.0f;

        for (int i = 0; i < width; i++) {
            float curr_val = A[row * width + i];
            float curr_max = fmaxf(curr_val, max_val);
            float curr_sum = sum_val * expf(max_val - curr_max) + expf(curr_val - curr_max);
            max_val = curr_max;
            sum_val = curr_sum;
        }

        for (int i = 0; i < width; i++) {
            B[row * width + i] = expf(A[row * width + i] - max_val) / sum_val;
        }
    }
}

void cudaSoftmax(float *input, float *output, int height, int width) {
    const dim3 blockDim(32, 1, 1);
    const dim3 gridDim((height + 32 - 1)/32, 1, 1);

    softmax_kernel<<<gridDim, blockDim>>>(input, output, height, width);
    cudaDeviceSynchronize();
}
