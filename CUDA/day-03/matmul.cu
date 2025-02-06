%%cuda
// Matrix multiplication of square Matrices of size Width
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void printMatrix(float *matrix, int Width) {
    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            printf("%.2f ", matrix[i * Width + j]);
        }
        printf("\n");
    }
    printf("\n");
}


__global__ void matrixMultKernel(float *M, float *N, float *P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}


void matrixMult(float *M_h, float *N_h, float *P_h, int Width) {
    int size = Width * Width * sizeof(float);
    float *M_d, *N_d, *P_d;

    // Part 1: Allocate device memory for M, N and P
    // copy M, N to deivce memory
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

    cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads
    // to perform matrix multiplication
    dim3 dimGrid(ceil(Width / 4.0), ceil(Width / 4.0), 1);
    dim3 dimBlock(4, 4, 1);
    matrixMultKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);

    cudaError_t err = cudaGetLastError(); // Capture error if kernel launch fails
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // Ensure kernel execution completes before moving on


    // Part 3: Copy the result from the device to the host
    // Free the device memory
    cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}


int main() {
    int Width = 16;     // Size of square matrix
    int size = Width * Width * sizeof(float);

    // Allocate memory for host matrices
    float *M_h = (float *)malloc(size);
    float *N_h = (float *)malloc(size);
    float *P_h = (float *)malloc(size);

    // Initialize matrices M and N with random values
    srand(time(NULL));
    for (int i = 0; i < Width * Width; i++) {
        M_h[i] = (float)(rand() % 10);  // random val between 0 and 9
        N_h[i] = (float)(rand() % 10);
    }

    // Print matrices M and N
    printf("Matrix M:\n");
    printMatrix(M_h, Width);

    printf("Matrix N:\n");
    printMatrix(N_h, Width);

    // Matrix multiplication in CUDA
    matrixMult(M_h, N_h, P_h, Width);

    // Print result of matrix P
    printf("Matrix P:\n");
    printMatrix(P_h, Width);

    // Free host
    free(M_h);
    free(N_h);
    free(P_h);

    return 0;
}
