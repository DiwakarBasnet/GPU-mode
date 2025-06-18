#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SECTION_SIZE 16

__global__ void parallel_scan_kernel(float *X, float *Y, int N) {
	__shared__ float XY[SECTION_SIZE];
	int i = 4 * blockIdx.x * blockDim.x + threadIdx.x;
	for (int b = 0; b < blockDim.x; b++) {
		if (i + b * blockDim.x < N) {
			XY[i + b * blockDim.x] = X[i + b * blockDim.x];
		}
	}
	
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		int val = 0
		if (threadIdx.x >= stride) {
			val = XY[threadIdx.x - stride];
		}
		__syncthreads();
		XY[threadIdx.x] += val;
		__syncthreads();
	}

	for (unsigned int idx = 0; idx < blockDim.x; idx++) {
		__syncthreads();
		if (blockIdx.x > 0) {
			unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
			XY[index] += XY[idx + (blockDim.x - 1)];
		}
	}
	__syncthreads();
	for (int c = 0; c < blockDim.x; c++) {
		if (i + c * blockDim.x < N) {
			Y[i + c * blockDim.x] = XY[i + c * blockDim.x];
		}
	}
}

void parallel_scan(float *X_h, float *Y_h, int N) {
	int size = N * sizeof(float);
	float *X_d, *Y_d;

	// Allocate device memory
	cudaError_t err1 = cudaMalloc((void**)&X_d, size);
	if (err1 != cudaSuccess) {
		printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
	}

	cudaError_t err2 = cudaMalloc((void**)&Y_d, size);
	if (err2 != cudaSuccess) {
		printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
	}
	
	cudaMemcpy(X_d, X_h, size, cudaMemcpyHostToDevice);

	// Initialize cuda kernel
	dim3 dimBlock(SECTION_SIZE/4, 1, 1);
	dim3 dimGrid(1, 1, 1);

	parallel_scan_kernel<<<dimGrid, dimBlock>>>(X_d, Y_d, N);

	// Copy output from device to host
	cudaMemcpy(Y_h, Y_d, size, cudaMemcpyDeviceToHost);

	cudaFree(X_d);
	cudaFree(Y_d);
}

int main() {
	int N = 16;
	int size = N * sizeof(float);
	
	// Allocate host memory
	float *X_h = (float *)malloc(size);
	float *Y_h = (float *)malloc(size);
	
	printf("Input:\n");
	// Initialize input
	for (int i = 0; i < N; i++) {
		X_h = (float)(i);
		printf("%.2f ", X_h[i]);
	}
	
	parallel_scan(X_h, Y_h, N);

	// Output result
	printf("\nOutput:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", Y_h[i]);
	}

	free(X_h_);
	free(Y_h);

	return 0;
}
