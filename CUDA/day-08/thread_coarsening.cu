#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define TILE_WIDTH 8
#define COARSE_FACTOR 4

void printMatrix(float *matrix, int width) {
    for(int r = 0; r < width; ++r) {
        for(int c = 0; c < width; ++c) {
            printf("%.2f ", matrix[r * width + c])''
        }
        printf("\n");
    }
}

__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // Initialize Pvalue for all output elements
    float Pvalue[COARSE_FACTOR];
    for(int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }

    // Loop over the M and N tiles required to compute P element
    for(int ph = 0; ph < width/TILE_WIDTH; ++ph) {

        // Collaborative loading of M tile into shared memory
        Mds[ty][tx] = M[row*width + ph*TILE_WIDTH + tx];

        for(int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c*TILE_WIDTH;

            // Collaborative loading of N tile into shared memory
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*width + col];
            __syncthreads();

            for(int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for(int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c*TILE_WIDTH;
        P[row*width + col] = Pvalue[c];
    }
}

void matrixMul(float *M_h, float *N_h, float *P_h, int width) {
    int size = width * width * sizeof(float);
    float *M_d, *N_d, *P_d;

    // Part 1: Memory allocation in device
    // copy M and N from host to device
    cudaError_t err1 = cudaMalloc((void**)&M_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&N_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }
    cudaError_t err3 = cudaMalloc((void**)&P_d, size);
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    // Part 2: Kernel initialization
    size_t SM_size = TILE_WIDTH * TILE_WIDTH * 2 * sizeof(float);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((width + TILE_WIDTH - 1)/TILE_WIDTH, (width + TILE_WIDTH - 1)/TILE_WIDTH, 1);

    matrixMulKernel<<<dimGrid, dimBlock, SM_size>>>(M_d, N_d, P_d, width);

    // Part 3: Capture error if kernel launch fails
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA kernel lauch failed: %s\n", cudaGetErrorString(err))''
    }
    cudaDeviceSynchronize();

    // Part 4: Copy result from device to host
    // Free device memory
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int maint() {
    int width = 16;
    int size = width * width * sizeof(float);

    // Allocate memory for host matrices
    float *M_h = (float *)malloc(size);
    float *N_h = (float *)malloc(size);
    float *P_h = (float *)malloc(size);

    // Initialize matrices with random values
    srand(time(NULL));
    for(int i = 0; i < width * width; ++i) {
        M_h[i] = (float)(rand() % 10);
        N_h[i] = (float)(rand() % 10);
    }

    // Print matrices M and N
    printf("\nMatrix M:\n");
    printMatirx(M_h, width);

    printf("\nMatrix N:\n");
    printMatirx(N_h, width);

    // Matrix multiplication in CUDA
    matrixMul(M_h, N_h, P_h, width);

    // Print output matrix P
    printf("\nMatrix P:\n");
    printMatirx(P_h, width);

    // Free host memory
    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}