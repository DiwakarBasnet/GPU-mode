#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 64

__global__ void SharedMemorySumReductionKernel(float *input, float *output) {
    unsigned int t = threadIdx.x;
    __shared__ float input_shared[BLOCK_SIZE];
    input_shared[t] = input[t] + input[t + BLOCK_SIZE];
    for (unsigned int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_shared[t] += input_shared[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = input_shared[0];
    }
}

void SharedMemorySumReduction(float *input_h, float *output_h, int N) {
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

    // Launch kernel
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(1);

    SharedMemorySumReductionKernel<<<dimGrid, dimBlock>>>(input_d, output_d);

    // Check for kernel launch errors
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    // Copy result back to host
    cudaMemcpy(output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);
}

int main() {
    int N = 256;
    int size = N * sizeof(float);
    
    float *input_h = (float *)malloc(size);
    float *output_h = (float *)malloc(sizeof(float));

    // Initialize input array
    for (int i = 0; i < N; i++) {
        input_h[i] = i + 1;
    }

    // Perform reduction
    SharedMemorySumReduction(input_h, output_h, N);
    printf("Sum: %f\n", *output_h);

    free(input_h);
    free(output_h);
    
    return 0;
}