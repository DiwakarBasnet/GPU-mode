#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 256

__global__ void SegmentedSumReductionKernel(float *input, float *output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

void SegmentedSumReduction(float *input_h, float *output_h, int N) {
    int size = N * sizeof(float);
    float *input_d, *output_d;

    // Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&input_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&output_d, sizeof(float));
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);

    // Lauch kernel
    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid((N + BLOCK_DIM - 1)/BLOCK_DIM);

    SegmentedSumReductionKernel<<<dimGrid, dimBlock>>>(input_d, output_d);

    // Check for kernel launch errors
    cudaError_t err3 = cduaGetLastError();
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    // Copy result back to host
    cudaMemcpy(output_h, output_d, sizeof(flaot), cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);
}

int main() {
    int N = 1024;
    float *input_h = (float *)malloc(N * sizeof(float));
    float *output_h = (float *)malloc(sizeof(float));

    for (int i = 0; i < N; i++) {
        input_h[i] = (float)(rand() % 10);
    }

    *output_h = 0.0f;
    SegmentedSumReduction(input_h, output_h, N);
    printf("Segmented sum: %f\n", *output_h);
    free(input_h);
    free(output_h);
    return 0;
}

