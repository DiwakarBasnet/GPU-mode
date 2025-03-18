#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IN_TILE_WIDTH 16
#define OUT_TILE_WIDTH 12

#define c0 0.02f
#define c1 0.52f
#define c2 0.25f
#define c3 0.12f
#define c4 0.89f
#define c5 0.37f
#define c6 0.93f


__global__ void stencil_sweep_kernel(float *in, float *out, unsigned int N) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    int i = blockIdx.z * OUT_TILE_WIDTH + z - 1;
    int j = blockIdx.y * OUT_TILE_WIDTH + y - 1;
    int k = blockIdx.x * OUT_TILE_WIDTH + x - 1;

    __shared__ float in_s[IN_TILE_WIDTH][IN_TILE_WIDTH][IN_TILE_WIDTH];

    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[z][y][x] = in[i*N*N + j*N + k];
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (z >= 1 && z < IN_TILE_WIDTH - 1 &&
            y >= 1 && y < IN_TILE_WIDTH - 1 &&
            x >= 1 && x < IN_TILE_WIDTH - 1) {
                out[i*N*N + j*N + k] = c0 * in_s[z][y][x] +
                                       c1 * in_s[z][y][x - 1] +
                                       c2 * in_s[z][y][x + 1] +
                                       c3 * in_s[z][y - 1][x] +
                                       c4 * in_s[z][y + 1][x] +
                                       c5 * in_s[z - 1][y][x] +
                                       c6 * in_s[z + 1][y][x];
            }
    }
}

void stencil_sweep(float *in_h, float *out_h, unsigned int N) {
    int size = N * N * N * sizeof(float);

    float *in_d, *out_d;

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

    // Lauch stencil kernel
    dim3 dimGrid((N + IN_TILE_WIDTH - 1)/IN_TILE_WIDTH, (N + IN_TILE_WIDTH - 1)/IN_TILE_WIDTH, (N + IN_TILE_WIDTH - 1)/IN_TILE_WIDTH);
    dim3 dimBlock(IN_TILE_WIDTH, IN_TILE_WIDTH, IN_TILE_WIDTH);

    stencil_sweep_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);

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
        printf("##############################################");
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