#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-6

__global__ void layer_norm_kernel(
    float *A, float *B, int height, int width
) {
    __shared__ float smem[1024];

    int row = blockIdx.x;   // one block per row
    int col = threadIdx.x;

    if (row < height) {
        float *row_in = A + row * width;
        float *row_out = B + row * width;

        float lmean = 0.0f;
        float lvar = 0.0f;

        // local mean and variance
        for (int i = col; i < width; i += blockDim.x) {
            float a = row_in[i];
            lmean += a;
            lvar += a * a;
        }

        __syncthreads();
        smem[col] = lmean;
        __syncthreads();

        // global mean: using reduction
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (col < stride) {
                smem[col] += smem[col + stride];
            }
            __syncthreads();
        }

        float gmean = smem[0] / n;
        __syncthreads();

        // now we can store local squared sums in smem
        smem[col] = lvar;
        __syncthreads();

        // global variance: using reduction
        for (int stride = blockDim.x; stride > 0; stride /=2) {
            if (col < stride) {
                smem[col] += smem[col + stride];
            }
            __syncthreads();
        }
        float gvar = (smem[0]/n) - (gmean * gmean);
        float stddev = rsqrt(gvar + EPSILON);
        __syncthreads();

        // normalize and store outputs
        for (int i = col; i < width; i += blockDim.x) {
            row_out[i] = (row_in[i] - gmean) * stddev;
        }
    }
    else {
        return;
    }
}

void run_smem_ln(float *A_d, float *B_d, int height, int width) {
    dim3 blockDim(256);
    dim3 gridDim(height);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    cudaEventRecord(start);

    layer_norm_kernel<<<gridDim, blockDim>>>(A_d, B_d, height, width);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Elapsed time: %f ms\n", ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}