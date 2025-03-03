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
                
            }
        }
    }
}