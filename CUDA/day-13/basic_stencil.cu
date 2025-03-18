#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 8

#define c0 0.02f
#define c1 0.52f
#define c2 0.25f
#define c3 0.12f
#define c4 0.89f
#define c5 0.37f
#define c6 0.93f

// Basic stencil sweep kernel
__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        out[i*N*N + j*N + k] = c0*in[i*N*N + j*N + k] +
                               c1*in[i*N*N + j*N + (k-1)] + c2*in[i*N*N + j*N + (k+1)] +
                               c3*in[i*N*N + (j-1)*N + k] + c4*in[i*N*N + (j+1)*N + k] +
                               c5*in[(i-1)*N*N + j*N + k] + c6*in[(i+1)*N*N + j*N + k];
    }
}

void stencil_sweep(float *in_h, float *out_h, unsigned int N) {
    int size = N * N * N * sizeof(float);
    float *in_d, *out_d;

    // Allocate memory in device
    cudaError_t err1 = cudaMalloc((void**)&in_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&out_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);

    // Kernel initialization
    dim3 dimGrid((N + TILE_WIDTH - 1)/TILE_WIDTH, (N + TILE_WIDTH - 1)/TILE_WIDTH, (N + TILE_WIDTH - 1)/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);

    stencil_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);

    // Check kernel execution for error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s error in kernel execution", cudaGetErrorString(err));
    }

    // Copy kernel output from device to host
    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);
}

void printStencil(float *stencil, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                printf("%.2f ", stencil[i*N*N + j*N + k]);
            }
            printf("\n");
        }
        printf("\n----------------------------------------------\n");
    }
}

int main() {
    int N = 8;
    int size = N * N * N * sizeof(float);

    float *in_h = (float *)malloc(size);
    float *out_h = (float *)malloc(size);

    // Initialization
    for (int i = 0; i < N*N*N; i++) {
        in_h[i] = (float)(rand() % 8);
    }

    printf("\nOrignial Input:\n");
    printStencil(in_h, N);

    stencil_sweep(in_h, out_h, N);

    printf("\nOutput stencil\n");
    printStencil(out_h, N);

    free(in_h);
    free(out_h);

    return 0;
}