#include <stdio.h>

// Vector addition on host
void vec_add(float* a_h, float* b_h, float* c_h, int n) {
    for (int i = 0; i < n; i++) {
        c_h[i] = a_h[i] + b_h[i];
    }
}

// Vector addition on device (Kernel)
__global__ void vec_add_kernel(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Vector addition on device
__global__ void vec_add_device(float* a_h, float* b_h, float* c_h, int n) {
    int size = n * sizeof(float);
    float *a_d, *b_d, *c_d;

    // Part 1: Allocate device memory for a, b, c
    cudaError_t err = cudaMalloc((void**)&a_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&b_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&c_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
    // copy a, b to device memory
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads
    vec_add_kernel<<<ceil(n / 256.0), 256>>>(a_d, b_d, c_d, n);

    // Part 3: Copy the result from the device to the host
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(a_d);
    cuda_Free(b_d);
    cudaFree(c_d);
}

int main() {
    int N = 10;

    // Initialize the vectors
    float A[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float B[N] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float C[N];

    // Vector addition on host
    //vec_add(A, B, C, N);

    // Vector addition on device
    vec_add_device(A, B, C, N);
    for (int i = 0; i < N; i++) {
        printf("%f ", C[i]);
    }

    return 0;
}