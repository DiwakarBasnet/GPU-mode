%%cuda
#include <stdio.h>
#include <stdlib.h>

#define SECTION_SIZE 16

__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, unsigned int N) {
	__shared__ float XY[SECTION_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		XY[threadIdx.x] = X[i];
	} else {
		XY[threadIdx.x] = 0.0f;
	}
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		float temp;
		if (threadIdx.x >= stride) {
			temp = XY[threadIdx.x] + XY[threadIdx.x-stride];
		}
		__syncthreads();
		if (threadIdx.x >= stride) {
			XY[threadIdx.x] = temp;
		}
	}
	if (i < N) {
		Y[i] = XY[threadIdx.x];
	}
}

void Kogge_Stone_scan(float *X_h, float *Y_h, unsigned int N) {
	int size = N * sizeof(float);
	
	// Allocate device memory
	float *X_d, *Y_d;
	cudaError_t err1 = cudaMalloc((void**)&X_d, size);
	if (err1 != cudaSuccess) {
		printf("%s in %s at %d", cudaGetErrorString(err1), __FILE__, __LINE__);
	}
	cudaError_t err2 = cudaMalloc((void**)&Y_d, size);
	if (err2 != cudaSuccess) {
		printf("%s in %s at %d", cudaGetErrorString(err2), __FILE__, __LINE__);
	}

	cudaMemcpy(X_d, X_h, size, cudaMemcpyHostToDevice);
	
	// Initialize cuda kernel
	dim3 dimBlock(SECTION_SIZE, 1, 1);
	dim3 dimGrid((N + SECTION_SIZE - 1)/SECTION_SIZE, 1, 1);

	Kogge_Stone_scan_kernel<<<dimGrid, dimBlock>>>(X_d, Y_d, N);

	// Determine if kernel launched successfully
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error launching cuda kernel");
	}
	cudaDeviceSynchronize();

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

	// Initialize host memory
	for (int i = 0; i < N; i++) {
		X_h[i] = (float)(i);
	}

	// Prefix Scan
	Kogge_Stone_scan(X_h, Y_h, N);

	// Print result
	printf("\nInput matrix:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", X_h[i]);
	}

	printf("\nOutput matrix:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", Y_h[i]);
	}

	// Free memory
	free(X_h);
	free(Y_h);

	return 0;
}
