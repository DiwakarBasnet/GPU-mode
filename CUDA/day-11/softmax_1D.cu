#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 8

// Online softmax kernel for 1D array
__global__ void online_softmax_kernel(
    float *a, float *b, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 1. Compute max and sum of exp(x[idx] - max_val)
        float max_val = -INFINITY;
        float sum = 0.0f;

        for (int i = 0; i < n; i++) {
            float cur_val = a[idx + i];
            float max_cur = fmaxf(cur_val, max_val);
            float sum_cur = sum * expf(max_val - max_cur) + expf(cur_val - max_cur);
            max_val = max_cur;
            sum = sum_cur;
        }

        // 2. Compute softmax
        for (int i = 0; i < n; i++) {
            b[idx + i] = expf(a[idx + i] - max_val) / sum;
        }
    }
}

void online_softmax(
    float *a_h, float *b_h, int n
) {
    int size = n * sizeof(float);
    float *a_d, *b_d;

    // 1. Allocate memory on the device
    cudaError_t err1 = cudaMalloc((void**)&a_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&b_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }
    
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

    // 2. Launch the cuda kernel
    dim3 dimBlock(TILE_WIDTH, 1, 1);
    dim3 dimGrid((n + TILE_WIDTH - 1)/TILE_WIDTH, 1, 1);
    online_softmax_kernel<<<dimGrid, dimBlock>>>(a_d, b_d, n);

    // 3. Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // 4. Copy result back to host and free memory
    cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);
    
    cudaFree(a_d);
    cudaFree(b_d);
}

int main() {
    int n = 16;
    int size = n * sizeof(float);

    float *a_h = (float*)malloc(size);
    float *b_h = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < n; i++) {
        a_h[i] = (float)(rand() % 10);
    }

    // Calculate softmax
    online_softmax(a_h, b_h, n);

    // Print input and output
    printf("Input: ");
    for (int i = 0; i < n; i++) {
        printf("%f. ", a_h[i]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < n; i++) {
        printf("%f, ", b_h[i]);
    }

    free(a_h);
    free(b_h);
    
    return 0;
}