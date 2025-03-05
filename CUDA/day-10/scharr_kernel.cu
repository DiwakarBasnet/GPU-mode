#include <math.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define FILTER_RADIUS 1
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)
#define TILE_DIM 32

__constant__ float SCHARR_X[FILTER_SIZE][FILTER_SIZE];
__constant__ float SCHARR_Y[FILTER_SIZE][FILTER_SIZE];

__global__ void scharr_kernel(
    float *image,
    float *output,
    int width,
    int height
) {
    __shared__ float input_tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (row < height && col < width) {
        input_tile[threadIdx.y][threadIdx.x] = image[row * width + col];
    } else {
        input_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (row < height && col < width) {
        float sum_x = 0.0f;
        float sum_y = 0.0f;

        for (int f_row = 0; f_row < FILTER_RADIUS; f_row++) {
            for (int f_col = 0; f_col < FILTER_RADIUS; f_col++) {
                if ((int)threadIdx.x - FILTER_RADIUS + fCol >= 0 &&
                    (int)threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
                    (int)threadIdx.y - FILTER_RADIUS + fRow >= 0 &&
                    (int)threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM) {
                        sum_x += SCHARR_X[fRow][fCol] * input_tile[threadIdx.y - FILTER_RADIUS + fRow][threadIdx.x - FILTER_RADIUS + fCol];
                        sum_y += SCHARR_Y[fRow][fCol] * input_tile[threadIdx.y - FILTER_RADIUS + fRow][threadIdx.x - FILTER_RADIUS + fCol];
                    }
                else {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 &&
                        col - FILTER_RADIUS + fCol < width) {
                            sum_x += SCHARR_X[fRow][fCol] * image[(row - FILTER_RADIUS + fRow) * width + (col - FILTER_RADIUS + fCol)];
                            sum_y += SCHARR_Y[fRow][fCol] * image[(row - FILTER_RADIUS + fRow) * width + (col - FILTER_RADIUS + fCol)];
                        }
                }
            }
        }
        output[row * width + col] = sqrtf(sum_x * sum_x + sum_y * sum_y);
    }
}

void scharr_filters() {
    float scharr_x[FILTER_SIZE][FILTER_SIZE] = {
        {-3, 0, 3},
        {-10, 0, 10},
        {-3, 0, 3}
    };
    float scharr_y[FILTER_SIZE][FILTER_SIZE] = {
        {-3, -10, -3},
        {0, 0, 0},
        {3, 10, 3}
    };

    cudaMemcpyToSymbol(SCHARR_X, scharr_x, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMemcpyToSymbol(SCHARR_Y, scharr_y, FILTER_SIZE * FILTER_SIZE * sizeof(float));
}

torch::Tensor scharr_cuda_forward(torch::Tensor input) {
    input = input.contiguous();

    const int height = input.size(0);
    const int width = input.size(1);

    auto output = torch::zeros_like(input);

    const dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    const dim3 dimGrid((width + TILE_DIM -1)/TILE_DIM, (height + TILE_DIM - 1)/TILE_DIM, 1);

    scharr_filters();

    scharr_kernel<<<dimGrid, dimBlock>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        width,
        height
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error in scharr_cuda_forward: %s\n", cudaGetErrorString(err));
    }

    return output;
}