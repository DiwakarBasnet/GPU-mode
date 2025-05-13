#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#define NUM_SAMPLES 5
#define FEATURE_DIMENSION 6

void printMatrix(float *matrix, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf("%.3f ", matrix[i * col + j]);
		}
		printf("\n");
	}
}

// Kernel: Softmax
__global__ void softmaxKernel(float *scoreMatrix, float *softmaxMatrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < NUM_SAMPLES) {
		float maxScore = -1e30f;
		for (int col = 0; col < NUM_SAMPLES; ++col) {
			maxScore = fmaxf(maxScore, scoreMatrix[row * NUM_SAMPLES + col]);
		}
		float sumExp = 0.0f;
		for (int col = 0; col < NUM_SAMPLES; ++col) {
			softmaxMatrix[row * NUM_SAMPLES + col] = 
				expf(scoreMatrix[row * NUM_SAMPLES + col] - maxScore);
			sumExp += softmaxMatrix[row * NUM_SAMPLES + col];
		}
		for (int col = 0; col < NUM_SAMPLES; ++col) {
			softmaxMatrix[row * NUM_SAMPLES + col] /= sumExp;
		}
	}
}

// Kernel: QK^T
__global__ void computeScoreKernel(float *queryMatrix, float *keyMatrix, float *scoreMatrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < NUM_SAMPLES && col < NUM_SAMPLES) {
		float score = 0.0f;
		for (int d = 0; d < FEATURE_DIMENSION; ++d) {
			score += queryMatrix[row * FEATURE_DIMENSION + d] *
				keyMatrix[col * FEATURE_DIMENSION + d];
		}
		scoreMatrix[row * NUM_SAMPLES + col] = score / sqrtf(static_cast<float>(FEATURE_DIMENSION));
	}
}

// Kernel: Output = Softmax(QK^T) * V
__global__ void computeOutputKernel(float * softmaxMatrix, float *valueMatrix, float *outputMatrix) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < NUM_SAMPLES && col < FEATURE_DIMENSION) {
		float result = 0.0f;
		for (int k = 0; k < NUM_SAMPLES; ++k) {
			result += softmaxMatrix[row * NUM_SAMPLES + k] *
				valueMatrix[k * FEATURE_DIMENSION + col];
		}
		outputMatrix[row * FEATURE_DIMENSION + col] = result;
	}
}

void computeAttention(float *queryMatrix_h, float *keyMatrix_h, float *valueMatrix_h, float *attnMatrix_h) {
	float size = NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float);
	float size_temp = NUM_SAMPLES * NUM_SAMPLES * sizeof(float);

	float *queryMatrix, *keyMatrix, *valueMatrix, *attnMatrix, *scoreMatrix, *softmaxMatrix;

	// Device memory allocation
	cudaMalloc((void**)&queryMatrix, size);
	cudaMalloc((void**)&keyMatrix, size);
	cudaMalloc((void**)&valueMatrix, size);
	cudaMalloc((void**)&attnMatrix, size);
	cudaMalloc((void**)&scoreMatrix, size_temp);
	cudaMalloc((void**)&softmaxMatrix, size_temp);

	cudaMemcpy(queryMatrix, queryMatrix_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(keyMatrix, keyMatrix_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(valueMatrix, valueMatrix_h, size, cudaMemcpyHostToDevice);
	
	// Kernel initializations
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((NUM_SAMPLES+blockDim.x-1)/blockDim.x, (NUM_SAMPLES+blockDim.y-1)/blockDim.y, 1);
	computeScoreKernel<<<gridDim, blockDim>>>(queryMatrix, keyMatrix, scoreMatrix);
	cudaDeviceSynchronize();

	dim3 softmaxBlockDim(16, 16, 1);
	dim3 softmaxGridDim((NUM_SAMPLES+softmaxBlockDim.x-1)/softmaxBlockDim.x, (NUM_SAMPLES+softmaxBlockDim.y-1)/softmaxBlockDim.y, 1);
	softmaxKernel<<<softmaxGridDim, softmaxBlockDim>>>(scoreMatrix, softmaxMatrix);
	cudaDeviceSynchronize();

	dim3 outputBlockDim(16, 16, 1);
	dim3 outputGridDim((NUM_SAMPLES+outputBlockDim.x-1)/outputBlockDim.x, (NUM_SAMPLES+outputBlockDim.y-1)/outputBlockDim.y, 1);
	computeOutputKernel<<<outputGridDim, outputBlockDim>>>(softmaxMatrix, valueMatrix, attnMatrix);
	cudaDeviceSynchronize();

	// Copy output from device to host
	cudaMemcpy(attnMatrix_h, attnMatrix, size, cudaMemcpyDeviceToHost);

	cudaFree(queryMatrix);
	cudaFree(keyMatrix);
	cudaFree(valueMatrix);
	cudaFree(attnMatrix);
	cudaFree(scoreMatrix);
	cudaFree(softmaxMatrix);
}

int main() {
	int size = NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float);

	float *queryMatrix = (float *)malloc(size);
	float *keyMatrix = (float *)malloc(size);
	float *valueMatrix = (float *)malloc(size);
	float *attnMatrix = (float *)malloc(size);

	// Initialize matrix
	for (int i = 0; i < NUM_SAMPLES * FEATURE_DIMENSION; i++) {
		queryMatrix[i] = (float)(rand() % 50);
		keyMatrix[i] = (float)(rand() % 50);
		valueMatrix[i] = (float)(rand() % 50);
	}

	printf("\nQuery:\n");
	printMatrix(queryMatrix, NUM_SAMPLES, FEATURE_DIMENSION);

	printf("\nKey:\n");
	printMatrix(keyMatrix, NUM_SAMPLES, FEATURE_DIMENSION);

	printf("\nValue\n");
	printMatrix(valueMatrix, NUM_SAMPLES, FEATURE_DIMENSION);

	// Attention calculation
	computeAttention(queryMatrix, keyMatrix, valueMatrix, attnMatrix);

	// Print attention matrix
	printf("\nAttention matrix;\:\n");
	printMatrix(attnMatrix, NUM_SAMPLES, FEATURE_DIMENSION);

	// Free memory
	free(queryMatrix);
	free(keyMatrix);
	free(valueMatrix);
	free(attnMatrix);

	return 0;
}
