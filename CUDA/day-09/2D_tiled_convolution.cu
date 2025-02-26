#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

void printMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

void printFilter(float **filter) {
    for (int i = 0; i < 2*FILTER_RADIUS+1; i++) {
        for (int j = 0; j < 2*FILTER_RADIUS+1; j++) {
            printf("%.2f ", filter[i][j]);
        }
        printf("\n");
    }
}

// __host__ int calculate_appropriate_SM_usage(cudaDeviceProp device_prop) {
//     // 1. Shared memory constraints
//     size_t sm_per_block = device_prop.sharedMemPerBlock;
//     size_t max_tile_elements = sm_per_block / sizeof(float);
//     // Tile width according to shared memory constraints
//     int tile_width_sm = (int)floor(sqrt(max_tile_elements));

//     // 2. Threads per block constraints
//     size_t max_threads_per_block = device_prop.maxThreadsPerBlock;
//     // Tile width according to threads per block constraints
//     int tile_width_threads = (int)floor(sqrt(max_threads_per_block));

//     // 3. Warp size constraints
//     int warp_size = device_prop.warpSize;

//     // 4. Final tile width
//     tile_width = min(min(tile_width_sm, tile_width_threads), warp_size);
//     // Ensure tile width is a multiple of warp size
//     tile_width = (tile_width / warp_size) * warp_size;

//     return tile_width;
// }

__global__ void convolution_tiled_2D_const_mem_kernel(
    float *N,   // Input image
    float *P,   // Output image
    int width,  // Width of input
    int height  // Height of input
) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Load the input tile to shared memory
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >=0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    // Turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += F_c[fRow][fCol] * N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

void convolution_tiled_2D_const_mem(
    float *N_h, float *P_h, float **F_h, int width, int height
) {
    int size = width * height * sizeof(float);
    int size_f = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(float);
    float *N_d, *P_d;

    // 1. Allocate device memory
    cudaError_t err1 = cudaMalloc((void**)&N_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&P_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F_c, F_h, size_f);

    // 2. Initialize cuda kernel
    dim3 dimGrid((width + IN_TILE_DIM - 1)/IN_TILE_DIM, (height + IN_TILE_DIM -1)/IN_TILE_DIM, 1);
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);

    convolution_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);

    // 3. Check if the kernel executed without errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    cudaDeviceSynchronize();

    // 4. Copy the result back to host
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    int width = 24;
    int height = 24;
    int size = width * height * sizeof(float);

    float *N_h = (float*)malloc(size);
    float *P_h = (float*)malloc(size);
    
    // Allocate memory for 2d array floating pointer
    float **F_h = (float**)malloc((2*FILTER_RADIUS+1) * sizeof(float*));

    // Randomly initialize the input image
    for (int i = 0; i < width * height; i++) {
        N_h[i] = (float)(rand() % 256);
    }
    // Randomly initialize the filter matrix
    for (int i = 0; i < 2*FILTER_RADIUS+1; i++) {
        for (int j = 0; j< 2*FILTER_RADIUS+1; j++) {
            F_h[i][j] = (float)(rand() % 10);
        }
    }

    // Convolution
    convolution_tiled_2D_const_mem(N_h, P_h, F_h, width, height);

    // Print the matrices
    printf("\nMatrix N:\n");
    printMatrix(N_h, width, height);
    printf("\nFilter F:\n");
    printFilter(F_h);
    printf("\nMatrix P:\n");
    printMatrix(P_h, width, height);
    
    free(N_h);
    free(P_h);
    free(F_h);

    return 0;
}
