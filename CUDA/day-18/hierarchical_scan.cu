#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SECTION_SIZE 16
#define NUM_BLOCKS 4
#define BLOCK_SIZE (SECTION_SIZE / NUM_BLOCKS)

__global__ void hierarchical_scan_kernel(float *X, float *Y, float *block_sum) {
	__shared__ float XY[SECTION_SIZE];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (idx < SECTION_SIZE) XY[tid] = X[idx];
	__syncthreads();

	// Local blockwise prefix scan
	for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
		float val = 0.0f;
		if (tid >= offset) {
			val = XY[tid - offset];
		}
		__syncthreads();

		XY[tid] += val;
		__syncthreads();
	}

	// Write block prefix sum to global memory
	Y[idx] = XY[tid];

	// Write last thread in block to block_sum
	if (tid == BLOCK_SIZE - 1) {
		block_sum[blockIdx.x] = XY[tid];
	}
}

__global__ void scan_block_sum(float *block_sum) {
    __shared__ float temp[NUM_BLOCKS];
    int tid = threadIdx.x;

    temp[tid] = block_sum[tid];
    __syncthreads();

    for (int offset = 1; offset < NUM_BLOCKS; offset *= 2) {
        if (tid >= offset) {
            temp[tid] += temp[tid - offset];
        }
        __syncthreads();
    }
    block_sum[tid] = temp[tid];
}

__global__ void apply_offsets(float *Y, float *block_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && idx < SECTION_SIZE) {
        Y[idx] += block_sum[blockIdx.x - 1];
    }
}

void hierarchical_scan(float *X_h, float *Y_h, int N) {
	int size = N * sizeof(float);
	int block_size = NUM_BLOCKS * sizeof(float);
	float *X_d, *Y_d, *block_sum;

	// Allocate device memory
	cudaError_t err1 = cudaMalloc((void**)&X_d, size);
	if (err1 != cudaSuccess) {
		printf("%s, %s in %d", cudaGetErrorString(err1), __FILE__, __LINE__);
	}

	cudaError_t err2 = cudaMalloc((void**)&Y_d, size);
	if (err2 != cudaSuccess) {
		printf("%s, %s in %d", cudaGetErrorString(err2), __FILE__, __LINE__);
	}

	cudaError_t err3 = cudaMalloc((void**)&block_sum, block_size);
	if (err3 != cudaSuccess) {
		printf("%s, %s in %d", cudaGetErrorString(err3), __FILE__, __LINE__);
	}

	cudaMemcpy(X_d, X_h, size, cudaMemcpyHostToDevice);

	// Initialize kernels
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(NUM_BLOCKS, 1, 1);

	hierarchical_scan_kernel<<<dimGrid, dimBlock>>>(X_d, Y_d, block_sum);
 
  scan_block_sum<<<1, NUM_BLOCKS>>>(block_sum);
 
  apply_offsets<<<NUM_BLOCKS, BLOCK_SIZE>>>(Y_d, block_sum);
 
  // Copy output from device to host and free space
	cudaMemcpy(Y_h, Y_d, size, cudaMemcpyDeviceToHost); 
	
	cudaFree(X_d);
	cudaFree(Y_d);
	cudaFree(block_sum);
}

int main() {
	int N = 16;
	int size = N * sizeof(float);

	float *X_h = (float *)malloc(size);
	float *Y_h = (float *)malloc(size);
	
	printf("Input:\n");
	for (int i = 0; i < N; i++) {
		X_h[i] = (float)i;
		printf("%.2f ", X_h[i]);
	}
	
	hierarchical_scan(X_h, Y_h, N);

	printf("\nOutput:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", Y_h[i]);
	}

	free(X_h);
	free(Y_h);

	return 0;
}
