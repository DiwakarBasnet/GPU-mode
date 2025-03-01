#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_DIM 32
#define FILTER_RADIUS 2

__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

void printMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

__global__ void conv_2D_cached_kernel(
    float *N, float *P, int width, int height
) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Loading input tile
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculate output elements
    // turning off the threads at the edges of the block
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                if (threadIdx.x-FILTER_RADIUS+fCol >= 0 &&
                    threadIdx.x-FILTER_RADIUS+fCol < TILE_DIM &&
                    threadIdx.y-FILTER_RADIUS+fRow >= 0 &&
                    threadIdx.y-FILTER_RADIUS+fRow < TILE_DIM) {
                        Pvalue += F_c[fRow][fCol] * N_s[threadIdx.y - FILTER_RADIUS + fRow][threadIdx.x - FILTER_RADIUS + fCol];
                    }
                else {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 &&
                        col - FILTER_RADIUS + fCol < width) {
                            Pvalue += F_c[fRow][fCol] * N[(row - FILTER_RADIUS + fRow)*width + (col - FILTER_RADIUS + fCol)];
                        }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}

void conv_2D_cached(float *N_h, float *P_h, float *F_h, int width, int height) {
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

    // 2. Kernel launch code
    dim3 dimGrid((width + TILE_DIM - 1)/TILE_DIM, (height + TILE_DIM - 1)/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    conv_2D_cached_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);

    // 3. Check if the kernel executes properly
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
    }
    cudaDeviceSynchronize();

    // 4. Copy back the result
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    int width = 64;
    int height = 64;
    int size = width * height * sizeof(float);
    int size_f = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);

    // Allocate host memory
    float *N_h = (float*)malloc(size);
    float *P_h = (float*)malloc(size);
    float *F_h = (float*)malloc(size_f);

    // Initialize host memory with random values
    for (int i = 0; i < width * height; i++) {
        N_h[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1); i++) {
        F_h[i] = (float)(rand() % 4);
    }

    // Convolution
    conv_2D_cached(N_h, P_h, F_h, width, height);

    // Print the result
    printf("\nInput Matrix:\n");
    printMatrix(N_h, width, height);
    printf("\nFilter Matrix:\n");
    printMatrix(F_h, 2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1);
    printf("\nOutput Matrix:\n");
    printMatrix(P_h, width, height);

    free(N_h);
    free(P_h);
    free(F_h);

    return 0;
}