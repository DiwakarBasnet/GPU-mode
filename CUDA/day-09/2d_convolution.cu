#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 8

void printMatrix(float *matrix, int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            printf("%.2f ", matrix[r * width + c]);
        }
        printf("\n");
    }
}

__global__ void convolution_2D_basic_kernel(
    float *N,   // pointer to input array
    float *F,   // pointer to filter 
    float *P,   // pointer to output array
    int r,      // radius of square filter
    int width,  // width of input and output arrays
    int height  // height of input and output arrays
) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * (2*r+1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

void convolution_2D_basic(
    float *N_h, float *F_h, float *P_h, int r, int width, int height
) {
    float *N_d, *F_d, *P_d;
    int size = width * height * sizeof(float);
    int size_f = (2*r+1) * (2*r+1) * sizeof(float);

    // 1. Allocate memory in GPU
    cudaError_t err1 = cudaMalloc((void**)&N_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&F_d, size_f);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }cudaError_t err3 = cudaMalloc((void**)&P_d, size);
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, size_f, cudaMemcpyHostToDevice);

    // 2. Initialize cuda kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((width + TILE_WIDTH - 1)/TILE_WIDTH, (height + TILE_WIDTH - 1)/TILE_WIDTH, 1);

    convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(N_d, F_d, P_d, r, width, height);

    // 3. Check if the kernel executed successfully
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel didn't lauch successfully");
    }

    cudaDeviceSynchronize();
    
    // 4. Copy output result from device to host
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(N_d);
    cudaFree(F_d);
    cudaFree(P_d);
}

int main() {
    int width = 24;
    int height = 32;
    int r = 2;
    int size = width * height * sizeof(float);
    int size_f = (2*r+1) * (2*r+1) * sizeof(float);

    float *N_h = (float *)malloc(size);
    float *F_h = (float *)malloc(size_f);
    float *P_h = (float *)malloc(size);

    // Random initialization of elements for N
    for (int i = 0; i < width * height; i++) {
        N_h[i] = (float)(rand() % 10);
    }
    // Random initialization of filter
    for (int i = 0; i < (2*r+1) * (2*r+1); i++) {
        F_h[i] = (float)(rand() % 5);
    }

    // Function call
    convolution_2D_basic(N_h, F_h, P_h, r, width, height);

    // Print matrices
    printf("\nMatrix N:\n");
    printMatrix(N_h, width, height);

    printf("\nFilter matrix F:\n");
    printMatrix(F_h, 2*r+1, 2*r+1);

    printf("\nMatrix P:\n");
    printMatrix(P_h, width, height);

    free(N_h);
    free(F_h);
    free(P_h);

    return 0;
}