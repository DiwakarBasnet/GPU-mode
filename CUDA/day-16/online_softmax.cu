%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void OnlineSoftmaxKernel(float *input, float *output, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height) {
        float x_max = -INFINITY;
        float norm = 0.0f;

        // Pass 1: Calculation of max and norm in single pass
        for (int col = 0; col < width; col++) {
           int i = row * width + col;
           if (x_max < input[i]) {
               norm = norm * expf(x_max - input[i]);
               x_max = input[i];
           }
           norm += expf(input[i] - x_max);
        }

        // Pass 2: Softmax calculation
        for (int col = 0; col < width; col++) {
            int i = row * width + col;
            output[i] = expf(input[i] - x_max) / norm;
        }
    }
}

void OnlineSoftmax(float *input_h, float *output_h, int width, int height) {
    int size = width * height * sizeof(float);
    float *input_d, *output_d;

    // Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&input_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&output_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    
    // Initialize kernel
    dim3 dimBlock(32);
    dim3 dimGrid((height + dimBlock.x - 1)/dimBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    OnlineSoftmaxKernel<<<dimGrid, dimBlock>>>(input_d, output_d, width, height);

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

void printMatrix(float *matrix, int height, int width) {
  for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
          printf("%f ", matrix[r * width + c]);
      }
      printf("\n");
  }
}

int main() {
    int height = 8;
    int width = 8;
    int size = height * width * sizeof(float);

    float *input_h = (float *)malloc(size);
    float *output_h = (float *)malloc(size);

    for (int i = 0; i < height * width; i ++) {
        input_h[i] = (float)(rand() % 10);
    }

    printf("Original input:\n");
    printMatrix(input_h, height, width);

    OnlineSoftmax(input_h, output_h, height, width);

    printf("\nSoftmax output\n");
    printMatrix(output_h, height, width);

    free(input_h);
    free(output_h);

    return 0;
}
