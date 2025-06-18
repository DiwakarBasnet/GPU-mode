%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SECTION_SIZE 16
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (SECTION_SIZE / THREADS_PER_BLOCK)

__global__ void parallel_scan_kernel(float *X, float *Y, int N) {
	__shared__ float XY[SECTION_SIZE];
  
  const int base = blockIdx.x * SECTION_SIZE;

  // Load input into shared memory
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      int idx = base + i * blockDim.x + threadIdx.x;
      XY[i * blockDim.x + threadIdx.x] = (idx < N) ? X[idx] : 0.0f;
  }
  __syncthreads();

  // Inclusive scan using Hillis-Steele
  for (int offset = 1; offset < SECTION_SIZE; offset <<= 1) {
      float tmp[ELEMS_PER_THREAD];

      #pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
          int pos = i * blockDim.x + threadIdx.x;
          tmp[i] = (pos >= offset) ? XY[pos - offset] : 0.0f;
      }
      __syncthreads();

      #pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; i++) {
          int pos = i * blockDim.x + threadIdx.x;
          XY[pos] += tmp[i];
      }
      __syncthreads();
  }

	#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      int idx = base + i * blockDim.x + threadIdx.x;
      if (idx < N) {
          Y[idx] =  XY[i * blockDim.x + threadIdx.x];
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
		X_h[i] = (float)(i);
		printf("%.2f ", X_h[i]);
	}
	
	parallel_scan(X_h, Y_h, N);

	// Output result
	printf("\nOutput:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", Y_h[i]);
	}

	free(X_h);
	free(Y_h);

	return 0;
}

