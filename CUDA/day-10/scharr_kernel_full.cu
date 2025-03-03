#include <stdio.h>
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

void print_matrix(float *matrix, int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            printf("%.2f ", matrix[row * width + col]);
        }
        printf("\n");
    }
}

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
                if (threadIdx.x-FILTER_RADIUS+fCol >= 0 &&
                    threadIdx.x-FILTER_RADIUS+fCol < TILE_DIM &&
                    threadIdx.y-FILTER_RADIUS+fRow >= 0 &&
                    threadIdx.y-FILTER_RADIUS+fRow < TILE_DIM) {
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

void scharr_2D(
    float *image_h,
    float *output_h,
    float scarr_x[FILTER_SIZE][FILTER_SIZE],
    float scarr_y[FILTER_SIZE][FILTER_SIZE],
    int width,
    int height
) {
    float *image_d, *output_d;
    int size = width * height * sizeof(float);

    // 1. Allocate memory on device
    cudaError_t err1 = cudaMalloc(&image_d, width * height * sizeof(float));
    if err1 != cudaSuccess {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc(&output_d, width * height * sizeof(float));
    if err2 != cudaSuccess {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(image_d, input_h, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(SCHARR_X, scarr_x, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    cudaMemcpyToSymbol(SCHARR_Y, scarr_y, FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // 2. Kernel launch
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    scharr_kernel<<<dimGrid, dimBlock>>>(image_d, output_d, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 3. Copy result back to host
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);

    cudaFree(image_d);
    cudaFree(output_d);

    printf("GPU time elapsed: %f ms\n", milliseconds);

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

int main() {
    int height = 1280;
    int width = 720;
    int size = height * width * sizeof(float);

    float *image_h = (float *)malloc(size);
    float *output_h = (float *)malloc(size);

    // Initialize image with random values
    for (int i = 0; i < height * width; i++) {
        image_h[i] = (float)(rand() % 10);
    }

    // Initialize scharr filters
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

    // Convolution function
    scharr_2D(image_h, output_h, scharr_x, scharr_y, width, height);

    print_matrix(output_h, width, height);

    free(image_h);
    free(output_h);

    return 0;
}