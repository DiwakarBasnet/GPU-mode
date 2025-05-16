#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_SAMPLES 5
#define FEATURE_DIMENSION 6
#define TILE_WIDTH 16

// Print utility
void printMatrix(const float* matrix, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%.3f ", matrix[r * cols + c]);
        }
        printf("\n");
    }
}

// Kernel: compute Q * K^T (scores)
__global__ void scoreKernel(
    const float* __restrict__ query,
    const float* __restrict__ keyT,
    float* __restrict__ score) {
    __shared__ float sharedQ[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedK[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int col = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;
    float acc = 0.0f;

    int phases = (FEATURE_DIMENSION + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int p = 0; p < phases; ++p) {
        int qCol = p * TILE_WIDTH + tx;
        int kRow = p * TILE_WIDTH + ty;

        // Load Q tile
        if (row < NUM_SAMPLES && qCol < FEATURE_DIMENSION)
            sharedQ[ty][tx] = query[row * FEATURE_DIMENSION + qCol];
        else
            sharedQ[ty][tx] = 0.0f;
        // Load K^T tile
        if (col < NUM_SAMPLES && kRow < FEATURE_DIMENSION)
            sharedK[ty][tx] = keyT[kRow * NUM_SAMPLES + col];
        else
            sharedK[ty][tx] = 0.0f;
        __syncthreads();

        // Dot-product
        for (int i = 0; i < TILE_WIDTH; ++i) {
            acc += sharedQ[ty][i] * sharedK[i][tx];
        }
        __syncthreads();
    }

    if (row < NUM_SAMPLES && col < NUM_SAMPLES) {
        score[row * NUM_SAMPLES + col] = acc / sqrtf((float)FEATURE_DIMENSION);
    }
}

// Kernel: row-wise softmax
__global__ void softmaxKernel(
    const float* __restrict__ score,
    float* __restrict__ softmax) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < NUM_SAMPLES) {
        float maxv = -1e30f;
        for (int c = 0; c < NUM_SAMPLES; ++c)
            maxv = fmaxf(maxv, score[row * NUM_SAMPLES + c]);
        float sum = 0.0f;
        for (int c = 0; c < NUM_SAMPLES; ++c) {
            float e = expf(score[row * NUM_SAMPLES + c] - maxv);
            softmax[row * NUM_SAMPLES + c] = e;
            sum += e;
        }
        for (int c = 0; c < NUM_SAMPLES; ++c)
            softmax[row * NUM_SAMPLES + c] /= sum;
    }
}

// Kernel: softmax * V
__global__ void outputKernel(
    const float* __restrict__ softmax,
    const float* __restrict__ value,
    float* __restrict__ output) {
    __shared__ float sharedS[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedV[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int col = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;
    float acc = 0.0f;

    int phases = (NUM_SAMPLES + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int p = 0; p < phases; ++p) {
        int sCol = p * TILE_WIDTH + tx;
        int vRow = p * TILE_WIDTH + ty;

        // Load softmax tile
        if (row < NUM_SAMPLES && sCol < NUM_SAMPLES)
            sharedS[ty][tx] = softmax[row * NUM_SAMPLES + sCol];
        else
            sharedS[ty][tx] = 0.0f;
        // Load V tile
        if (vRow < NUM_SAMPLES && col < FEATURE_DIMENSION)
            sharedV[ty][tx] = value[vRow * FEATURE_DIMENSION + col];
        else
            sharedV[ty][tx] = 0.0f;
        __syncthreads();

        // Dot-product
        for (int i = 0; i < TILE_WIDTH; ++i) {
            acc += sharedS[ty][i] * sharedV[i][tx];
        }
        __syncthreads();
    }

    if (row < NUM_SAMPLES && col < FEATURE_DIMENSION) {
        output[row * FEATURE_DIMENSION + col] = acc;
    }
}

// Host helper: transpose key
void transposeKey(const float* key, float* keyT) {
    for (int r = 0; r < NUM_SAMPLES; ++r)
        for (int c = 0; c < FEATURE_DIMENSION; ++c)
            keyT[c * NUM_SAMPLES + r] = key[r * FEATURE_DIMENSION + c];
}

int main() {
    size_t qSize = NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float);
    size_t kTSize = NUM_SAMPLES * FEATURE_DIMENSION * sizeof(float);
    size_t sSize = NUM_SAMPLES * NUM_SAMPLES * sizeof(float);

    // Host allocations
    float *hQ = (float*)malloc(qSize);
    float *hK = (float*)malloc(qSize);
    float *hV = (float*)malloc(qSize);
    float *hKT = (float*)malloc(kTSize);
    float *hScore = (float*)malloc(sSize);
    float *hSoftmax = (float*)malloc(sSize);
    float *hOut = (float*)malloc(qSize);

    // Random init
    for (int i = 0; i < NUM_SAMPLES * FEATURE_DIMENSION; ++i) {
        hQ[i] = rand() % 50;
        hK[i] = rand() % 50;
        hV[i] = rand() % 50;
    }

    printf("\nQuery:\n"); printMatrix(hQ, NUM_SAMPLES, FEATURE_DIMENSION);
    printf("\nKey:\n");   printMatrix(hK, NUM_SAMPLES, FEATURE_DIMENSION);
    printf("\nValue:\n"); printMatrix(hV, NUM_SAMPLES, FEATURE_DIMENSION);

    // Transpose key on host
    transposeKey(hK, hKT);

    // Device allocations
    float *dQ, *dKT, *dV, *dScore, *dSoftmax, *dOut;
    cudaMalloc(&dQ, qSize);
    cudaMalloc(&dKT, kTSize);
    cudaMalloc(&dV, qSize);
    cudaMalloc(&dScore, sSize);
    cudaMalloc(&dSoftmax, sSize);
    cudaMalloc(&dOut, qSize);

    // Copy to device
    cudaMemcpy(dQ, hQ, qSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dKT, hKT, kTSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, qSize, cudaMemcpyHostToDevice);

    // Launch score kernel
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 gridScore((NUM_SAMPLES+TILE_WIDTH-1)/TILE_WIDTH,
                   (NUM_SAMPLES+TILE_WIDTH-1)/TILE_WIDTH);
    scoreKernel<<<gridScore, block>>>(dQ, dKT, dScore);
    cudaDeviceSynchronize();

    // Softmax kernel
    dim3 gridSm((NUM_SAMPLES+TILE_WIDTH-1)/TILE_WIDTH, 1);
    softmaxKernel<<<gridSm, block>>>(dScore, dSoftmax);
    cudaDeviceSynchronize();

    // Output kernel
    dim3 gridOut((FEATURE_DIMENSION+TILE_WIDTH-1)/TILE_WIDTH,
                 (NUM_SAMPLES+TILE_WIDTH-1)/TILE_WIDTH);
    outputKernel<<<gridOut, block>>>(dSoftmax, dV, dOut);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(hOut, dOut, qSize, cudaMemcpyDeviceToHost);

    printf("\nAttention Output:\n");
    printMatrix(hOut, NUM_SAMPLES, FEATURE_DIMENSION);

    // Cleanup
    free(hQ); free(hK); free(hV); free(hKT);
    free(hScore); free(hSoftmax); free(hOut);
    cudaFree(dQ); cudaFree(dKT); cudaFree(dV);
    cudaFree(dScore); cudaFree(dSoftmax); cudaFree(dOut);

    return 0;
}

