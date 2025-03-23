#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Kernel computes histogram for letters 'a' to 'z'
__global__ void histo_kernel(const char *data, unsigned int length, unsigned int *histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            // Each thread atomically adds to the corresponding bin
            atomicAdd(&histo[alphabet_position/4], 1);
        }
    }
}

void histo(char *data_h, unsigned int length, unsigned int *histo_h) {
    int data_size = length * sizeof(char);
    int histo_size = 7 * sizeof(unsigned int);
    char *data_d;
    unsigned int *histo_d;

    // Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&data_d, data_size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaError_t err2 = cudaMalloc((void**)&histo_d, histo_size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Initialize device histogram to zero
    cudaError_t err3 = cudaMemset(histo_d, 0, histo_size);
    if (err3 != cudaSuccess) {
        printf("%s in %s at %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Copy input data from host to device
    cudaMemcpy(data_d, data_h, data_size, cudaMemcpyHostToDevice);

    // Kernel execution configuration
    dim3 dimBlock(32);
    dim3 dimGrid((length + dimBlock.x - 1) / dimBlock.x);

    histo_kernel<<<dimGrid, dimBlock>>>(data_d, length, histo_d);

    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // Copy the histogram from device to host
    cudaMemcpy(histo_h, histo_d, histo_size, cudaMemcpyDeviceToHost);

    cudaFree(data_d);
    cudaFree(histo_d);
}

int main() {
    // Example input string (should be lowercase letters for this example)
    char data[] = "this is an example of a cuda histogram computation";
    unsigned int length = strlen(data);

    unsigned int histo_host[7] = {0};

    // Compute histogram on the GPU
    histo(data, length, histo_host);

    // Print the histogram
    for (int i = 0; i < 7; i++) {
        printf("b%d: %u\n", i, histo_host[i]);
    }

    return 0;
}
