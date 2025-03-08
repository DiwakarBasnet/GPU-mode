%%cuda
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 8

void print_matrix(float *matrix, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

__global__ void online_2d_softmax_kernel(
    float *A, float *B, int height, int width
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height) {
        float max = A[row * width];
        float sum = 0.0f;
        
        for (int i = 0; i < width; i++) {
            float curr_val = A[row * width + i];
            float curr_max = fmaxf(curr_val, max);
            float curr_sum = sum * expf(max - curr_max) + expf(curr_val - curr_max);
            max = curr_max;
            sum = curr_sum;
        }

        // Compute softmax
        for (int i = 0; i < width; i++) {
            B[row * width + i] = expf(A[row * width + i] - max) / sum;
        }
    }
}

void online_2d_softmax(
    float *A_h, float *B_h, int height, int width
) {
    int size = height * width * sizeof(float);
    float *A_d, *B_d;

    // 1. Allocate memory on the device
    cudaError_t err1 = cudaMalloc((void**)&A_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&B_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    
    // 2. Launch the cuda kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((height + TILE_WIDTH - 1)/TILE_WIDTH, (width + TILE_WIDTH - 1)/TILE_WIDTH, 1);
    online_2d_softmax_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, height, width);

    // 3. Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // 4. Copy the result back to the host
    cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
}

int main() {
    int height = 8;
    int width = 8;
    int size = height * width * sizeof(float);

    float *A_h = (float*)malloc(size);
    float *B_h = (float*)malloc(size);

    // Randomly initialize A_h
    for (int i = 0; i < height * width; i++) {
        A_h[i] = (float)(rand() % 10);
    }

    // Calculate softmax
    online_2d_softmax(A_h, B_h, height, width);

    // Print input and output
    printf("Input matrix:\n");
    print_matrix(A_h, height, width);
    printf("\nOutput matrix:\n");
    print_matrix(B_h, height, width);

    free(A_h);
    free(B_h);

    return 0;
}