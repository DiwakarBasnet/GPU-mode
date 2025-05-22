#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#define NUM_SAMPLES 5
#define FEATURE_DIMENSION 6

void printMatrix(float *matrix, int row, int col) {
    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            printf("%.3f ", matrix[r * col + c]);
        }
        printf("\n");
    }
}

// Kernel: Attention Score (x = QK^T)
__global__ void attention_score_kernel(
    float *Q, float *K, float *x
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < NUM_SAMPLES && col < NUM_SAMPLES) {
        float sum = 0.0f;
        for (int i = 0; i < FEATURE_DIMENSION; i++) {
            sum += Q[row * FEATURE_DIMENSION + i] * K[col * FEATURE_DIMENSION + i];
        }
        x[row * NUM_SAMPLES + col] = sum;
    }
}

// Kernel: Flash Attention
__global__ void flash_attention_kernel(
    float *x, float *V, float *O
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < NUM_SAMPLES && col < FEATURE_DIMENSION) {
        float m = -INFINITY;
        float d = 0.0f;
        float o = 0.0f;

        for (int i = 0; i < NUM_SAMPLES; i++){
            float x_val = x[row * NUM_SAMPLES + i];
            float m_prev = m;
            float d_prev = d;

            // Compute running max and denominator
            m = fmaxf(m_prev, x_val);
            d = (d_prev * expf(m_prev - m)) + expf(x_val - m);

            // Compute output
            float v_val = V[i * FEATURE_DIMENSION + col];
            o = o * ((d_prev * expf(m_prev - m)) / d) + (expf(x_val- m) / d) * v_val;
        }
        O[row * FEATURE_DIMENSION + col] = o;
    }
}

void computeFlashAttention(
    float *Q, float *K, float *V, float *O
) {
    float *d_Q, *d_K, *d_V, *d_x, *d_O;
    size_t size_1 = NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float);
    size_t size_2 = NUM_SAMPLES * NUM_SAMPLES * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_Q, size_1);
    cudaMalloc((void**)&d_K, size_1);
    cudaMalloc((void**)&d_V, size_1);
    cudaMalloc((void**)&d_x, size_2);
    cudaMalloc((void**)&d_O, size_1);

    // Copy data from host to device
    cudaMemcpy(d_Q, Q, size_1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, size_1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, size_1, cudaMemcpyHostToDevice);

    // Kernel launch for attention score
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((NUM_SAMPLES + blockDim.x - 1)/blockDim.x, (NUM_SAMPLES + blockDim.y - 1)/blockDim.y, 1);
    attention_score_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_x);
    cudaDeviceSynchronize();

    // Kernel launch for flash attention
    dim3 blockDim2(16, 16, 1);
    dim3 gridDim2((NUM_SAMPLES + blockDim2.x - 1)/blockDim2.x, (NUM_SAMPLES + blockDim2.y - 1)/blockDim2.y, 1);
    flash_attention_kernel<<<gridDim2, blockDim2>>>(d_x, d_V, d_O);
    cudaDeviceSynchronize();

    // Copy Output from device to host
    cudaMemcpy(O, d_O, size_1, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_x);
    cudaFree(d_O);
}

int main() {
    float size = FEATURE_DIMENSION * NUM_SAMPLES * sizeof(float);
    float *Q = (float *)malloc(size);
    float *K = (float *)malloc(size);
    float *V = (float *)malloc(size);
    float *O = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < NUM_SAMPLES * FEATURE_DIMENSION; i++) {
        Q[i] = rand() % 50;
        K[i] = rand() % 50;
        V[i] = rand() % 50;
    }
    printf("\nQuery:\n"); printMatrix(Q, NUM_SAMPLES, FEATURE_DIMENSION);
    printf("\nKey:\n");   printMatrix(K, NUM_SAMPLES, FEATURE_DIMENSION);
    printf("\nValue:\n"); printMatrix(V, NUM_SAMPLES, FEATURE_DIMENSION);

    // Compute Flash Attention
    computeFlashAttention(Q, K, V, O);
    printf("\nOutput:\n"); printMatrix(O, NUM_SAMPLES, FEATURE_DIMENSION);

    // Free host memory
    free(Q);
    free(K);
    free(V);
    free(O);

    return 0;
}
