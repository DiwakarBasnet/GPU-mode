#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define TILE_WIDTH 16

void printMatrix(float *matrix, int Width) {
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%.2f ", matrix[i * Width + j]);
        }
        printf("\n");
    }
}

__global__ void matrixMulKernel(float *M, float *N, float *P, int Width){

    // scope of sma is blocks so 1 version of Mds and Nds will be created for each block
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];   // shared mem arrays (sma)
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // 1 version of bx, by, tx and ty will be created for each thread 
    // and will reside in registers that are accessible by the thread
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    // Once thread ends, the values of these variables cease to exist

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    // Strip-mining (break long loops into phases)
     // Each iteration corresponds to one phase of calculation
    for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph) {

        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph*TILE_WIDTH + tx) < Width) {
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];    // each phase uses one tile of M and one tile of N elements
        }
        else Mds[ty][tx] = 0.0f;
        if ((ph*TILE_WIDTH + ty) < Width && Col < Width) {
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        }
        else Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    if (Row < Width) && (Col < Width) {
        P[Row * Width + Col] = Pvalue;
    }
}

void matrixMul(float *M_h, float *N_h, float *P_h, int Width) {
    int size = Width * Width * sizeof(float);
    float *M_d, *N_d, *P_d;

    // Part 1: Allocate device memory for M, N and P
    // Copy M, N from host to device
    cudaError_t err1 = cudaMalloc((void**)&M_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&N_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }
    cudaError_t err3 = cudaMalloc((void**)&P_d, size);
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    // Part 2: Initialize kernel
    dim3 dimGrid((Width + 32 - 1)/32, (Width + 32 - 1)/32, 1);
    dim3 dimBlock(32, 32, 1);
    matrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);

    // Part 3: Capture error if kernel launch fails
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();    // Ensures kernel execution completes before moving on

    // Part 4: Copy result from device to host
    // Free device memory
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    int Width = 64;
    int size = Width * Width * sizeof(float);

    // Allocate memory for host matrices
    float *M_h = (float *)malloc(size);
    float *N_h = (float *)malloc(size);
    float *P_h = (float *)malloc(size);

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < Width * Width; i++) {
        M_h[i] = (float)(rand() % 10);  // random values between 0 and 9
        N_h[i] = (float)(rand() % 10);
    }

    // Print matrices M and N
    printf("Matrix M:\n");
    printMatrix(M_h, Width);

    printf("Matrix N:\n");
    printMatrix(N_h, Width);

    // Matrix multiplication in CUDA
    matrixMul(M_h, N_h, P_h, Width);

    // Print matrix multiplication output P
    printf("Matrix P:\n");
    printMatrix(P_h, Width);

    // Free host memory
    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}