#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define TILE_WIDTH 4

void printMatrix(float *matrix, int width, int height) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

__global__ void generalTiledMatmulKernel(
    float *M, float *N, float *P, int j, int k, int l
) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < ceil(k/(float)TILE_WIDTH); ++ph) {
        if ((Row < j) && ((ph * TILE_WIDTH + tx) < k)) {
            Mds[ty][tx] = M[Row * k + ph * TILE_WIDTH + tx];
        }
        else Mds[ty][tx] = 0.0f;
        if (((ph * TILE_WIDTH + ty) < k) && (Col < l)) {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty)* l + Col];
        }
        else Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int kp = 0; kp < TILE_WIDTH; ++kp) {
            Pvalue += Mds[ty][kp] * Nds[kp][tx];
        }
        __syncthreads();
    }
    if ((Row < j) && (Col < l)) {
        P[Row * l + Col] = Pvalue;
    }
}

void generalTiledMatmul(
    float *M_h, float *N_h, float *P_h, int j, int k, int l
) {
    int sizeM = j * k * sizeof(float);
    int sizeN = k * l * sizeof(float);
    int sizeP = l * j * sizeof(float);
    float *M_d, *N_d, *P_d;

    // Part 1: Allocate device memory for M, N and P
    // Copy M, N from host to device
    cudaError_t err1 = cudaMalloc((void**)&M_d, sizeM);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&N_d, sizeN);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }
    cudaError_t err3 = cudaMalloc((void**)&P_d, sizeP);
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    cudaMemcpy(M_d, M_h, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, sizeN, cudaMemcpyHostToDevice);

    // Part 2: Initialize kernel
    dim3 dimGrid((l + TILE_WIDTH - 1)/TILE_WIDTH, (k + TILE_WIDTH - 1)/TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    generalTiledMatmulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, j, k, l);

    // Part 3: Capture error if kernel launch fails
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // Part 4: Copy result from device to host
    // Free device memory
    cudaMemcpy(P_h, P_d, sizeP, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    int j = 6, k = 8, l = 5;
    int sizeM = j * k * sizeof(float);
    int sizeN = k * l * sizeof(float);
    int sizeP = l * j * sizeof(float);

    // Allocate memory for host matrices
    float *M_h = (float *)malloc(sizeM);
    float *N_h = (float *)malloc(sizeN);
    float *P_h = (float *)malloc(sizeP);

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < j * k; i++) {
        M_h[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < k * l; i++) {
        N_h[i] = (float)(rand() % 10);
    }

    // Print matrices M and N
    printf("\nMatrix M:\n");
    printMatrix(M_h, k, j);

    printf("\nMatrix N:\n");
    printMatrix(N_h, l, k);

    // Matrix multiplication in CUDA
    generalTiledMatmul(M_h, N_h, P_h, j, k, l);

    // Print matrix multiplication output P
    printf("\nMatrix P:\n");
    printMatrix(P_h, l, j);

    // Free host memory
    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}