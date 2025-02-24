#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 8

void printMatrix(float *vector, int width) {
    for (int i = 0; i < width; i++) {
        printf("%.2f ", vector[i]);
    }
}

__global__ void convolution_1D_basic_kernel(
    float *A,   // Input vector pointer
    float *B,   // Output vector pointer
    float *F,   // 1D filter vector pointer
    int r,      // radius of filter
    int width  // width of vector
) {
    int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0.0f;
    for (int i = 0; i < 2*r+1; i++) {
        int inIdx = outIdx - r + i;
        if (inIdx >= 0 && inIdx < width) {
            Pvalue += A[inIdx] * F[i];
        }
    }
    B[outIdx] = Pvalue;
}

void convolution_1D_basic(
    float *A_h, float *B_h, float *F_h, int r, int width
) {
    float *A_d, *B_d, *F_d;
    int size = width * sizeof(float);
    int size_f = (2 * r + 1) * sizeof(float);

    // 1. Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&A_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&B_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }
    cudaError_t err3 = cudaMalloc((void**)&F_d, size_f);
    if (err3 != cudaSuccess) {
        printf("%s in %s at %d", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, size_f, cudaMemcpyHostToDevice);

    // 2. Initialize cuda kernel
    dim3 dimBlock(TILE_WIDTH, 1, 1);
    dim3 dimGrid((width + TILE_WIDTH - 1)/TILE_WIDTH, 1, 1);

    convolution_1D_basic_kernel<<<dimBlock, dimGrid>>>(A_d, B_d, F_d, r, width);

    // 3. Determine if kernel launched successfully
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error launching cuda kernel");
    }
    cudaDeviceSynchronize();

    // 4. Copy output from device to host
    cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(F_d);
}

int main() {
    int width = 26;
    int r = 3;
    int size = width * sizeof(float);
    int size_f = (2 * r + 1) * sizeof(float);

    // Allocate host memory
    float *A_h = (float *)malloc(size);
    float *B_h = (float *)malloc(size);
    float *F_h = (float *)malloc(size_f);

    // Initialize host memory
    for (int i = 0; i < width; i++) {
        A_h[i] = (float)(rand() % 16);
    }
    for (int i = 0; i < 2*r+1; i++) {
        F_h[i] = (float)(rand() % 4);
    }

    // Convolution
    convolution_1D_basic(A_h, B_h, F_h, r, width);

    // Print result
    printf("\nMatrix A:\n");
    printMatrix(A_h, width);

    printf("\nMatrix F:\n");
    printMatrix(F_h, 2*r+1);

    printf("\nMatrix B:\n");
    printMatrix(B_h, width);

    // Free memory
    free(A_h);
    free(B_h);
    free(F_h);

    return 0;
}