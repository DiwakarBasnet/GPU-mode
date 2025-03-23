#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_BINS 7
#define CFACTOR  3

__global__ void histo_private_kernel(char *data, unsigned int length, unsigned int *histo) {
    // Initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[binIdx] = 0u;
    }
    __syncthreads();
    // Histogram
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid * CFACTOR; i < min((tid+1)*CFACTOR, length); ++i) {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo_s[alphabet_position/4]), 1);
        }
    }
    __syncthreads();
    // Commit to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[binIdx];
        if (binValue > 0) {
            atomicAdd(&(histo[binIdx]), binValue);
        }
    }
}

void histo_private(char *data_h, unsigned int length, unsigned int *histo_h) {
    int size_data = length * sizeof(char);
    int size_histo = NUM_BINS * sizeof(unsigned int);

    char *data_d;
    unsigned int *histo_d;

    // Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&data_d, size_data);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&histo_d, size_histo);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(data_d, data_h, size_data, cudaMemcpyHostToDevice);

    // Kernel execution
    dim3 dimBlock(32);
    dim3 dimGrid((length + blockDim.x - 1)/blockDim.x);

    histo_private_kernel<<<dimGrid, dimBlock>>>(data_d, length, histo_d);

    // Check if kernel executed successfully
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    // Copy result from device to host
    cudaMemcpy(histo_h, histo_d, size_histo, cudaMemcpyDeviceToHost);

    cudaFree(histo_d);
    cudaFree(data_d);
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