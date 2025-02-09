%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


void printMatrix(float *matrix, int num_rows, int num_cols) {
    printf("\n");
    for (int i  = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            printf("%.2f ", matrix[i * num_cols + j]);
        }
        printf("\n");
    }
}


__global__ void matVecKernel(
    float *A, float *B, float *C, int dimension
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dimension) {
        for(int k = 0; k < dimension; k++) {
            A[row] += B[row * dimension + k] * C[k];
        }
    }
}


void matVec(
    float *A_h, float *B_h, float *C_h, int dimension
) {
    int vec_size = dimension * sizeof(float);
    int mat_size = dimension * dimension * sizeof(float);

    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device memory for A, B and C
    // copy B and C to deivce memory
    cudaError_t err1 = cudaMalloc((void**)&A_d, vec_size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&B_d, mat_size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }
    cudaError_t err3 = cudaMalloc((void**)&C_d, vec_size);
    if (err3 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err3), __FILE__, __LINE__);
    }

    cudaMemcpy(B_d, B_h, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, vec_size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads
    // to perform matrix-vector multiplication
    dim3 dimGrid((dimension + 2 - 1)/2, (dimension + 2 - 1)/2, 1);
    dim3 dimBlock(2, 1, 1);

    matVecKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, dimension);

    // Part 3: Copy the result from the device to the host
    // Free the device memory
    cudaMemcpy(A_h, A_d, vec_size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main() {
    int dimension = 7;
    int mat_size = dimension * dimension * sizeof(float);
    int vec_size = dimension * sizeof(float);

    // Allocate host memory for matrix B and vector A and C
    float *A_h = (float *)malloc(vec_size);
    float *B_h = (float *)malloc(mat_size);
    float *C_h = (float *)malloc(vec_size);

    // Generate matrix B and vector C
    srand(time(NULL));
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            int offset = i * dimension + j;
          B_h[offset] = (float)(rand() % 9);    // [0, 9]
        }
    }
    for (int j = 0; j < dimension; j++) {
        C_h[j] = (float)(rand() % 9);
    }

    // Print matrix B and vector C
    printf("\nMatrix B");
    printMatrix(B_h, dimension, dimension);
    printf("\nVector C");
    printMatrix(C_h, dimension, 1);

    // Initialize kernel
    matVec(A_h, B_h, C_h, dimension);

    // Print output vector A
    printf("\nVector A");
    printMatrix(A_h, dimension, 1);

    // Free host
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}