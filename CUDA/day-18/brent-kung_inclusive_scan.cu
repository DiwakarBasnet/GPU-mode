%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SECTION_SIZE 16

__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int N) {
	__shared__ float XY[SECTION_SIZE];
	unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) XY[threadIdx.x] = X[i];
	if (i + blockDim.x < N) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
	for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
		if (index < SECTION_SIZE) {
			XY[index] += XY[index - stride];
		}
	}
	for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
		__syncthreads();
		unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < SECTION_SIZE) {
			XY[index + stride] += XY[index];
		}
	}
	__syncthreads();
	if (i < N) Y[i] = XY[threadIdx.x];
	if (i + blockDim.x < N) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

void Brent_Kung_scan(float *X_h, float *Y_h, unsigned int N) {
	float size = N * sizeof(float);
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
	cudaMemcpy(Y_d, Y_h, size, cudaMemcpyHostToDevice);

	// Initialize kernel
	dim3 dimBlock(8, 1, 1);
	dim3 dimGrid((dimBlock.x + N - 1)/dimBlock.x, 1, 1);
	Brent_Kung_scan_kernel<<<dimGrid, dimBlock>>>(X_d, Y_d, N);

	// Copy output to host memory
	cudaMemcpy(Y_h, Y_d, size, cudaMemcpyDeviceToHost);
	
	cudaFree(X_d);
	cudaFree(Y_d);
}

int main() {
	int N = 16;
	float size = N * sizeof(float);

	// Allocate host memory
	float *X_h = (float *)malloc(size);
	float *Y_h = (float *)malloc(size);
	
	printf("Input:\n");
	for (int i = 0; i < N; i++) {
		X_h[i] = (float)(i);
		printf("%.2f ", X_h[i]);
	}

	// Prefix Sum
	Brent_Kung_scan(X_h, Y_h, N);

	printf("\nOutput:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", Y_h[i]);
	}

	// Free memory
	free(X_h);
	free(Y_h);

	return 0;
}
