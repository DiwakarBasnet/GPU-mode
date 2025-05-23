#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void printMatrix(float *matrix, int Width) {
    for (int i = 0; i < Width; ++i) {
        for(int j = 0; j < Width; ++j) {
            printf("%.2f ", matrix[i * Width + j]);
        }
        printf("\n");
    }
}

__host__ int calculate_appropriate_SM_usage(cudaDeviceProp device_prop) {
    // 1. Shared memory constraints. 
    size_t shared_mem_per_block = device_prop.sharedMemPerBlock;
    // We need 2 tiles of size TILE_SIZE * TILE_SIZE. 
    size_t max_tile_elements = shared_mem_per_block / (2 * sizeof(float));
    // Calculate the tile size because tile_size * tile_size <= max_tile_elements. 
    int tile_size_from_shared = (int)floor(sqrt(max_tile_elements));

    // 2. Thread count constraints. 
    int max_threads_per_block = device_prop.maxThreadsPerBlock;
    // Each thread block has TILE_SIZE * TILE_SIZE threads. 
    int tile_size_from_threads = (int)floor(sqrt(max_threads_per_block));

    // 3. Warp size constraints.
    // The warp size is usually 32 for all NVIDIA GPUs.
    int warp_size = device_prop.warpSize;

    int tile_size = min(min(tile_size_from_shared, tile_size_from_threads), warp_size);

    // Ensure the tile size is a multiple of the warp size.
    tile_size = (tile_size / warp_size) * warp_size;

    // Print the GPU properties and the tile size.
    printf("Device name: %s\n", device_prop.name);
    printf("Shared memory per block: %zu bytes\n", shared_mem_per_block);
    printf("Max threads per block: %d\n", max_threads_per_block);
    printf("Warp size: %d\n", warp_size);
    printf("Tile size: %d\n", tile_size);
    printf("\n");

    return tile_size;
}

__global__ void dynamicTiledMatrixMulKernel(
    float *M, float *N, float *P, int Width, int tile_width
) {
    // Single shared memory array
    extern __shared__ float Mds_Nds[];

    float *Mds = (float *) Mds_Nds;
    float *Nds = (float *) (Mds_Nds + tile_width * tile_width);  // Starts right after Mds

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Row and Col of thread in P, which we are calculating
    int Row = by * tile_width + ty;
    int Col = bx * tile_width + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;
    for (int ph = 0; ph < ceil(Width/(float)tile_width); ++ph) {
        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph * tile_width + tx) < Width) {
            Mds[ty * tile_width + tx] = M[Row * Width + ph * tile_width + tx];
        }
        else Mds[ty * tile_width + tx] = 0.0f;
        if ((ph * tile_width + ty) < Width && (Col < Width)) {
            Nds[ty * tile_width + tx] = N[(ph * tile_width + ty) * Width + Col];
        }
        else Nds[ty * tile_width + tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < tile_width; ++k) {
            Pvalue += Mds[ty * tile_width + k] * Nds[k * tile_width + tx];
        }
        __syncthreads();
    }
    if ((Row < Width) && (Col < Width)) {
        P[Row * Width + Col] = Pvalue;
    }
}

void dynamicTiledMatrixMul(
    float *M_h, float *N_h, float *P_h, int Width
) {
    int size = Width * Width * sizeof(float);
    float *M_d, *N_d, *P_d;

    // Determine appropriate tile size
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);   // 0 means first GPU
    int tile_width = calculate_appropriate_SM_usage(devProp);

    // Part 1: Allocate device memory for M, N and P
    // Copy M, N from host to device
    cudaError_t err1 = cudaMalloc((void**)&M_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&N_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err2), __FILE__, __LINE__);
    }
    cudaError_t err3 = cudaMalloc((void**)&P_d, size);
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);

    // Part 2: Initialize kernel
    dim3 dimGrid((Width + tile_width - 1)/tile_width, (Width + tile_width - 1)/tile_width, 1);
    dim3 dimBlock(tile_width, tile_width, 1);

    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);

    dynamicTiledMatrixMulKernel<<<dimGrid, dimBlock, shared_mem_size>>>(M_d, N_d, P_d, Width, tile_width);

    // Part 3: Capture error if kernel launch fails
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();    // Ensures kernel execution completes before moving on

    // Part 4: Copy result from device to host
    // Free device memory
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main() {
    int Width = 16;
    int size = Width * Width * sizeof(float);

    // Allocate memory for host matrices
    float *M_h = (float *)malloc(size);
    float *N_h = (float *)malloc(size);
    float *P_h = (float *)malloc(size);

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < Width * Width; ++i) {
        M_h[i] = (float)(rand() % 10);  // random values between 0 and 9
        N_h[i] = (float)(rand() % 10);
    }

    // Print matrices M and N
    printf("\nMatrix M:\n");
    printMatrix(M_h, Width);

    printf("\nMatrix N:\n");
    printMatrix(N_h, Width);

    // Matrix multiplication in CUDA
    dynamicTiledMatrixMul(M_h, N_h, P_h, Width);

    // Print matrix multiplication output P
    printf("\nMatrix P:\n");
    printMatrix(P_h, Width);

    // Free host memory
    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}