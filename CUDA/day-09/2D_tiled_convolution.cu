#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
#define TILE_WIDTH IN_TILE_DIM

__constant__ float F_c[(2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1)];

void printMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

__global__ void convolution_tiled_2D_const_mem_kernel(
    float *N,   // Input image
    float *P,   // Output image
    int width,  // Width of input
    int height  // Height of input
) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Load the input tile to shared memory
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >=0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    // Turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += F_c[fRow * (2*FILTER_RADIUS+1) + fCol] * N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

void convolution_tiled_2D_const_mem(
    float *N_h, float *P_h, float *F_h, int width, int height
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
    cudaMemcpyToSymbol(F_c, F_h, size_f);

    // 2. Initialize cuda kernel
    dim3 dimGrid((width + TILE_WIDTH - 1)/TILE_WIDTH, (height + TILE_WIDTH -1)/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    convolution_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);

    // 3. Check if the kernel executed without errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaDeviceSynchronize();

    // 4. Copy the result back to host
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    int width = 12;
    int height = 12;
    int size = width * height * sizeof(float);
    int size_f = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);

    float *N_h = (float*)malloc(size);
    float *P_h = (float*)malloc(size);
    float *F_h = (float*)malloc(size_f);

    // Randomly initialize the input image
    for (int i = 0; i < width * height; i++) {
        N_h[i] = (float)(rand() % 10);
    }
    // Randomly initialize the filter matrix
    for (int i = 0; i < (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1); i++) {
        F_h[i] = (float)(rand() % 4);
    }

    // Convolution
    convolution_tiled_2D_const_mem(N_h, P_h, F_h, width, height);

    // Print the matrices
    printf("\nMatrix N:\n");
    printMatrix(N_h, width, height);
    printf("\nFilter F:\n");
    printMatrix(F_h, 2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1);
    printf("\nMatrix P:\n");
    printMatrix(P_h, width, height);
    
    free(N_h);
    free(P_h);
    free(F_h);

    return 0;
}