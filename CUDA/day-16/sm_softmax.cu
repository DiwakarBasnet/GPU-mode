#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 8

__global__ void SoftmaxKernel(float *input, float *output, int width, int height) {
	extern __shared__ float smem[];

	int row = blockIdx.x;
	int tid = threadIdx.x;

	float *input_row = input + row * width;
	float *output_row = output + row * width;
	float local_max = -INFINITY;
	float local_norm = 0.0f;

	for (int col = 0; col < width; col++) {
		float x = input_row[col];

		if (x > local_max) {
			local_norm *= expf(local_max - x);
			local_max = x;
		}
		local_norm += expf(x - local_max);
	}
	__syncthreads();

	smem[tid] = local_max;
	__syncthreads();

	for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
		if (tid < stride) {
			smem[tid] = max(smem[tid], smem[tid + stride]);
		}
		__syncthreads();
	}

	float global_max = smem[0];
	__syncthreads();

	for (int i = tid; i < N; i += blockIdx.x) {
		output_row[i] = expf(input_row[i] - row_max)/row_norm;
	}
}

void Softmax(float *input_h, float *output_h, int width, int height) {
	int size = width * height * sizeof(float);
	float *input_d, *output_d;
	
	// Allocate device memory
	cudaError_t err1 = cudaMalloc((void**)&input_d, size);
	if (err1 != cudaSuccess) {
		printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
	}
	cudaError_t err2 = cudaMalloc((void**)&output_d, size);
	if (err2 != cudaSuccess) {
		printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
	}

	cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);

	// Initialize kernel
	dim3 dimBlock(1024);
	dim3 dimGrid((height + dimBlock.x - 1)/dimBlock.x);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float ms = 0.f;
	cudaEventRecord(start);

	SoftmaxKernel<<<dimGrid, dimBlock, BLOCK_DIM>>>(input_d, output_d, width, height);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	printf(">> Kernel execution time: %f ms\n", ms);

	// Check for kernel launch errors
	cudaError_t err3 = cudaGetLastError();
	if (err3 != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
	}

	// Copy result from device to host
	cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);

	cudaFree(input_d);
	cudaFree(output_d);
}

int main() {
    int height = 8;
    int width = 8;
    int size = height * width * sizeof(float);

    float *input_h = (float *)malloc(size);
    float *output_h = (float *)malloc(size);

    for (int i = 0; i < height * width; i ++) {
        input_h[i] = (float)(rand() % 10);
    }

    printf("Original input:\n");
    printMatrix(input_h, height, width);

    OnlineSoftmax(input_h, output_h, height, width);

    printf("\nSoftmax output\n");
    printMatrix(output_h, height, width);

    free(input_h);
    free(output_h);

    return 0;
}	
