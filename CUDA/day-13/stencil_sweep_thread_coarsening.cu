#include <stdio.h>
#include <stdlib.h>

#define OUT_TILE_DIM 30
#define IN_TILE_DIM 32

__constant__ float c0; 
__constant__ float c1; 
__constant__ float c2; 
__constant__ float c3; 
__constant__ float c4; 
__constant__ float c5; 
__constant__ float c6;

__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    if (iStart-1 >= 0 && iStart-1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1)*N*N + j*N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    }
    for (int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1)*N*N + j*N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1
                && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                    out[i*N*N + j*N + k] = c0*inCurr_s[threadIdx.y][threadIdx.x]
                                         + c1*inCurr_s[threadIdx.y][threadIdx.x-1]
                                         + c2*inCurr_s[threadIdx.y][threadIdx.x+1]
                                         + c3*inCurr_s[threadIdx.y-1][threadIdx.x]
                                         + c4*inCurr_s[threadIdx.y+1][threadIdx.x]
                                         + c5*inPrev_s[threadIdx.y][threadIdx.x]
                                         + c6*inNext_s[threadIdx.y][threadIdx.x];
                }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

void stencil_sweep(float *in_h, float *out_h, unsigned int N) {
    int size = N * N * N * sizeof(float);

    float *in_d, *out_d;

    float h_c0 = 0.02f;
    float h_c1 = 0.52f;
    float h_c2 = 0.25f;
    float h_c3 = 0.12f;
    float h_c4 = 0.89f;
    float h_c5 = 0.37f;
    float h_c6 = 0.93f;

    // Device memory allocation
    cudaError_t err1 = cudaMalloc((void**)&in_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&out_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);

    // Constant memory allocation
    cudaMemcpyToSymbol(c0, &h_c0, sizeof(float));
    cudaMemcpyToSymbol(c1, &h_c1, sizeof(float));
    cudaMemcpyToSymbol(c2, &h_c2, sizeof(float));
    cudaMemcpyToSymbol(c3, &h_c3, sizeof(float));
    cudaMemcpyToSymbol(c4, &h_c4, sizeof(float));
    cudaMemcpyToSymbol(c5, &h_c5, sizeof(float));
    cudaMemcpyToSymbol(c6, &h_c6, sizeof(float));

    // Launch stencil kernel
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 dimGrid((N + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM, (N + OUT_TILE_DIM - 1)/OUT_TILE_DIM);

    stencil_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);

    // Check kernel launch for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s while kernel launch", cudaGetErrorString(err));
    }

    // Copy kernel results from device to host
    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);
}

void printStencil(float *stencil, int N) {
    for (int z = 0; z < N; z++) {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                printf("%.2f ", stencil[z*N*N + y*N + x]);
            }
            printf("\n");
        }
        printf("###############################################");
    }
}

int main() {
    int N = 16;
    int size = N * N * N * sizeof(float);

    float *in_h = (float *)malloc(size);
    float *out_h = (float *)malloc(size);

    for (int i = 0; i < N*N*N; i++) {
        in_h[i] = (float)(rand() % 10);
    }

    printf("\nOriginal Stencil:\n");
    printStencil(in_h, N);

    stencil_sweep(in_h, out_h, N);

    printf("\nResult Stencil:\n");
    printStencil(out_h, N);

    free(in_h);
    free(out_h);

    return 0;
}