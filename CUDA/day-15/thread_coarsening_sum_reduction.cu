#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 16
#define COARSE_FACTOR 4

__global__ void CoarsenedSumReductionKernel(float *input, float *output) {
	__shared__ float input_s[BLOCK_DIM];
	unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;
	float sum = input[i];
	for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
		sum += input[i + tile*BLOCK_DIM];
	}
	input_s[t] = sum;
	for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (t < stride) {
			input_s[t] += input_s[t + stride];
		}
	}
	if (t == 0) {
		atomicAdd(output, input_s[0]);
	}
}

void CoarsenedSumReduction(float *input_h, float *output_h, int N) {
	int size = N * sizeof(float);
	float *input_d, *output_d;

	// Allocate device memory
	cudaError_t err1 = cudaMalloc((void**)&input_d, size);
	if (err1 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
	}
	cudaError_t err2 = cudaMalloc((void**)&output_d, sizeof(float));
	if (err2 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
	}

	cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
  cudaMemset(output_d, 0 ,sizeof(float));

	// Launch kernel
	dim3 dimBlock(BLOCK_DIM, 1, 1);
	dim3 dimGrid(1, 1, 1);

	CoarsenedSumReductionKernel<<<dimGrid, dimBlock>>>(input_d, output_d);

	// Check for kernel lauch errors
	cudaError_t err3 = cudaGetLastError();
	if (err3 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
	}

	// Copy output result into device
	cudaMemcpy(output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(input_d);
	cudaFree(output_d);
}

int main() {
	int N = 256;
	int size = N * sizeof(float);

	float *input_h = (float *)malloc(size);
	float *output_h = (float *)malloc(sizeof(float));

	for (int i = 0; i <N; i++) {
		input_h[i] = i + 1;
	}

	CoarsenedSumReduction(input_h, output_h, N);

	printf("Sum = %f", *output_h);

	free(input_h);
	free(output_h);

	return 0;
}
