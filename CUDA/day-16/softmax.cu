#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void SimpleSoftmaxKernel(float *input, float *output, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    // Calculate exponentials for all elements
    output[idx] = expf(input[idx]);

    // Wait for all threads to finish exponentiation
    __syncthreads();

    // Compute sum of exponents
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += output[i];
    }

    // Normalize Values
    output[idx] = output[idx] / sum;

}


void SimpleSoftmax(float *input_h, float *output_h, int N) {
    float *input_d, *output_d;
    int size = N * sizeof(float);

    // Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&input_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&output_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(N);
    dim3 dimGrid(1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    SimpleSoftmaxKernel<<<dimGrid, dimBlock>>>(input_d, output_d, N);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nTime taken: %f ms\n", milliseconds);

    // Check for errors
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    // Copy result back to host from device
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);
}

int main() {
    int N = 8;
    int size = N * sizeof(float);

    float *input_h = (float *)malloc(size);
    float *output_h = (float *)malloc(size);

    for (int i = 0; i < N; i ++) {
        input_h[i] = (float)(rand() % 10);
    }

    printf("Original input:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", input_h[i]);
    }

    SimpleSoftmax(input_h, output_h, N);

    printf("\nSoftmax output\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", output_h[i]);
    }

    free(input_h);
    free(output_h);

    return 0;
}