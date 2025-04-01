#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void ConvergentSumReductionKernel(float *input, float *output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

void ConvergentSumReduction(float *input_h, float *output_h, int N) {
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
    dim3 dimBlock(N / 2);
    dim3 dimGrid(1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    ConvergentSumReductionKernel<<<dimGrid, dimBlock>>>(input_d, output_d);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

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
    int N = 128;
    float *input_h = (float *)malloc(N * sizeof(float));
    float *output_h = (float *)malloc(sizeof(float));

    for (int i = 0; i < N; i++) {
        input_h[i] = i + 1;
    }

    ConvergentSumReduction(input_h, output_h, N);

    printf("\nSum: %f\n", *output_h);

    free(input_h);
    free(output_h);

    return 0;
}