#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void printMatrix(float *matrix, int Width) {
    for (int i = 0; i < Width; ++i) {
        for(int j = 0; j < Width; ++j) {
            printf("%.2f ", matrix[i * Width + j]);
        }
        printf("\n");
    }
}

__global__ void dynamicTiledMatrixMulKernel(
    float *M, float *N, float *P, int Width,
    int tile_width, unsigned Mds_sz, unsigned Nds_sz
) {
    // Single shared memory array
    extern __shared__ char float Mds_Nds[];

    float *Mds = (float *) Mds_Nds;
    float *Nds = (float *) (Mds_Nds + Mds_sz);  // Starts right after Mds

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Row and Col of thread in P, which we are calculating
    int Row = by * tile_width + ty;
    int Col = bx * tile_width + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < ceil(Width/(float)tile_width); ++ph) {
        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph * tile_width + tx) < Width) {
            Mds[ty * tile_width + tx] = M[Row * Width + ph * tile_width + tx];
        }
        else Mds[ty * tile_width + tx] = 0.0f;
        if ((ph * tile_width + ty) < Width && (Col < Width)) {
            Nds[ty * tile_width + tx] = N[(ph * tile_width + ty) * Width + Col];
        }
        else Nds[ty * tile_width + tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < tile_width; ++k) {
            Pvalue += Mds[ty * tile_width + k] * Nds[k * tile_width + tx];
        }
        __syncthreads();
    }
    if ((Row < Width) && (Col < Width)) {
        P[Row * Width + Col] = Pvalue;
    }
}

void dynamicTiledMatrixMul(
    float *M_h, float *N_h, float *P_h, int Width, int tile_width
) {
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
    dim3 dimGrid((Width + tile_width - 1)/tile_width, (Width + tile_width - 1)/tile_width, 1);
    dim3 dimBlock(tile_width, tile_width, 1);
    size_t size = 2 * tile_width * tile_width * sizeof(float);

    dynamicTiledMatrixMulKernel<<<dimGrid, dimBlock, size>>>(M_d, N_d, P_d, Width, tile_width, size/2, size/2);

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
    int tile_width = 16;

    // Allocate memory for host matrices
    float *M_h = (float *)malloc(size);
    float *N_h = (float *)malloc(size);
    float *P_h = (float *)malloc(size);

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < Widht * Width; ++i) {
        M_h[i] = (float)(rand() % 10);  // random values between 0 and 9
        N_h[i] = (float)(rand() % 10);
    }

    // Print matrices M and N
    printf("\nMatrix M:\n");
    printMatrix(M_h, Width);

    printf("\nMatrix N:\n");
    printMatrix(N_h, Width);

    // Matrix multiplication in CUDA
    dynamicTiledMatrixMul(M_h, N_h, P_h, Width);

    // Print matrix multiplication output P
    printf("\nMatrix P:\n");
    printMatrix(P_h, Widht);

    // Free host memory
    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}