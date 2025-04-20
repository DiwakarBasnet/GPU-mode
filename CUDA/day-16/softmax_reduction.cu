#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_DIM 8   // must be >= your matrix width

void printMatrix(const float *mat, int width, int height) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            printf("%f ", mat[r * width + c]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void softmaxKernel(const float *input, float *output, int width) {
    extern __shared__ float sdata[];  // size == BLOCK_DIM

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int idx = row * width + tid;

    // 1) Load or -inf if out of bounds
    float x = (tid < width) ? input[idx] : -INFINITY;
    sdata[tid] = x;

    // 2) Reduction for max
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
    }
    __syncthreads();
    float row_max = sdata[0];

    // 3) Compute exponentials
    float e = (tid < width) ? expf(x - row_max) : 0.0f;
    sdata[tid] = e;

    // 4) Reduction for sum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
    }
    __syncthreads();
    float row_sum = sdata[0];

    // 5) Write normalized result
    if (tid < width) {
        output[idx] = e / row_sum;
    }
}

void softmax(const float *h_input, float *h_output, int width, int height) {
    size_t bytes = width * height * sizeof(float);
    float *d_input = nullptr, *d_output = nullptr;

    cudaMalloc(&d_input,  bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch: one block per row, BLOCK_DIM threads, shared mem = BLOCK_DIM floats
    dim3 dimGrid(height);
    dim3 dimBlock(BLOCK_DIM);
    softmaxKernel<<<dimGrid, dimBlock, BLOCK_DIM * sizeof(float)>>>(
        d_input, d_output, width
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int width  = 8;
    const int height = 8;

    size_t bytes = width * height * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    // Fill with random [0,9]
    for (int i = 0; i < width * height; ++i) {
        h_in[i] = rand() % 10;
    }

    printf("Original matrix:\n");
    printMatrix(h_in, width, height);

    softmax(h_in, h_out, width, height);

    printf("Softmax matrix:\n");
    printMatrix(h_out, width, height);

    free(h_in);
    free(h_out);
    return 0;
}
