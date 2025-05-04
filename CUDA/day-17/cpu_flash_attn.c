#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_SAMPLES 2
#define FEATURE_DIMENSION 3

void printMatrix(float *matrix, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf("%f ", matrix[i * col + j]);
		}
		printf("\n");
	}
}

// CPU Implementation of Attention
void transposeMatrix(float *in_matrix, float *out_matrix, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			out_matrix[j * row + i] = in_matrix[i * col + j];
		}
	}
}

void computeAttentionCPU(float *query, float *key, float *value,
		float *attentionScores, float *output) {
	float *transposeKey = (float*)malloc(FEATURE_DIMENSION * NUM_SAMPLES * sizeof(float));
	transposeMatrix(key, transposeKey, NUM_SAMPLES, FEATURE_DIMENSION);

	float scalingFactor = 1.0f / sqrt((float)FEATURE_DIMENSION);

	// Compute attention scores
	for (int i = 0; i < NUM_SAMPLES; i++) {
		for (int j = 0; j < NUM_SAMPLES; j++) {
			for (int k = 0; k < FEATURE_DIMENSION; k++) {
				attentionScores[i * NUM_SAMPLES + j] += query[i * FEATURE_DIMENSION + k] * transposeKey[k * NUM_SAMPLES + j];
			}
			attentionScores[i * NUM_SAMPLES + j] *= scalingFactor;
		}
	}

	// Softmax row-wise
	for (int row = 0; row < NUM_SAMPLES; row++) {
		float maxScore = attentionScores[row * NUM_SAMPLES];
		for (int col = 1; col < NUM_SAMPLES; col++) {
			if (attentionScores[row * NUM_SAMPLES + col] > maxScore) {
				maxScore = attentionScores[row * NUM_SAMPLES + col];
			}
		}
		float sumExp = 0.0f;
		for (int col = 0; col < NUM_SAMPLES; col++) {
			attentionScores[row * NUM_SAMPLES + col] = exp(attentionScores[row * NUM_SAMPLES + col] - maxScore);
			sumExp += attentionScores[row * NUM_SAMPLES + col];
		}
		for (int col = 0; col < NUM_SAMPLES; col++) {
			attentionScores[row * NUM_SAMPLES + col] /= sumExp;
		}
	}

	// Multiply by value matrix
	for (int i = 0; i < NUM_SAMPLES; i++) {
		for (int j = 0; j < FEATURE_DIMENSION; j++) {
			for (int k = 0; k < NUM_SAMPLES; k++) {
				output[i * FEATURE_DIMENSION + j] += attentionScores[i * NUM_SAMPLES + k] * value[k * FEATURE_DIMENSION + j];
			}
		}
	}

	free(transposeKey);
}

int main() {
	float query[NUM_SAMPLES * FEATURE_DIMENSION] = {
		1.0f, 0.0f, -1.0f,
		0.5f, 0.5f, 0.5f
	};

	float key[NUM_SAMPLES * FEATURE_DIMENSION] = {
		1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f
	};
	
	float value[NUM_SAMPLES * FEATURE_DIMENSION] = {
		1.0f, 1.0f, 1.0f,
		2.0f, 2.0f, 2.0f
	};

	float* output = (float*)malloc(FEATURE_DIMENSION * NUM_SAMPLES * sizeof(float));
	float* attentionScores = (float*)malloc(NUM_SAMPLES * NUM_SAMPLES * sizeof(float));
	computeAttentionCPU(query, key, value, attentionScores, output);

	printMatrix(output, FEATURE_DIMENSION, NUM_SAMPLES);

	free(output);
	free(attentionScores);

	return 0;
}



