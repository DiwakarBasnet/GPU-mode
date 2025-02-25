#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 8
#define FILTER_RADIUS 2

__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

void printMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

__global__ void convolution_2D_const_mem_kernel(
    float *N,   // input image pointer
    float *P,   // output image pointer
    int width,  // image width
    int height  // image height
) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0.0f;
    for (int i = 0; i < 2*FILTER_RADIUS+1; i++) {
        for (int j = 0; j < 2*FILTER_RADIUS+1; j++) {
            int inRow = outRow - FILTER_RADIUS + i;
            int inCol = outCol - FILTER_RADIUS + j;
            if (inRow >=0 && inRow < height && inCol >=0 && inCol < width) {
                Pvalue += N[inRow * width + inCol] * F[i][j];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

void convolution_2D_const_mem(
    float  *N_h, float *P_h, // input and output images
    int width, int height,  // image width and height
    float *F_h
) {
    int size = width * height * sizeof(float);
    int size_f = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);
    float *N_d, *P_d;

    // 1. Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&N_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&P_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, F_h, size_f);

    // 2. Kernel launch code
    dim3 dimGrid((width + TILE_WIDTH - 1)/TILE_WIDTH, (height + TILE_WIDTH - 1)/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    convolution_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);

    // 3. Check if the kernel encountered any error
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("Error launching kernel: %s\n", cudaGetErrorString(err3));
    }
    cudaDeviceSynchronize();

    // 4. Copy the output from device to host
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    int width = 20;
    int height = 15;
    int size = width * height * sizeof(float);
    int size_f = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);

    float *N_h = (float*)malloc(size);
    float *P_h = (float*)malloc(size);
    float *F_h = (float*)malloc(size_f);


    // Random initialization of input
    for (int i = 0; i < width * height; i++) {
        N_h[i] = (float)(rand() % 256);
    }
    // Random initialization of filter
    for (int i = 0; i < (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1); i++) {
        F_h[i] = (float)(rand() % 10);
    }

    // Convolution
    convolution_2D_const_mem(N_h, P_h, width, height, F_h);

    // Print matrices
    printf("Input matrix N:\n");
    printf("====================================\n");
    printMatrix(N_h, width, height);
    
    printf("Filter matrix F:\n");
    printf("====================================\n");
    printMatrix(F_h, 2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1);

    printf("Output matrix P:\n");
    printf("====================================\n");
    printMatrix(P_h, width, height);

    free(N_h);
    free(P_h);

    return 0;
}