#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print_matrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

__global__ void row_matrix_mult_kernel(float *A, float *B, float *C, int N) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N) {
        for (int col = 0; col < N; col++) {
            // According to the question, we want each thread to produce
            // a row of the output matrix, which means that each thread/row
            // will loop through all the columns of the product matrix. 
            float c_sum = 0.0;

            for (int k = 0; k < N; k++) {
                c_sum += A[thread_id * N + k] * B[k * N + col];
            }
            C[thread_id * N + col] = c_sum;
        }
    }
}

__global__ void col_matrix_mult_kernel(float *A, float *B, float *C, int N) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N) {
        for (int row = 0; row < N; row++) {
            // According to the question, we want each thread to produce
            // a column of the output matrix, which means that each thread/column
            // will loop through all the rows of the product matrix.
            float c_sum = 0.0;

            for (int k = 0; k < N; k++) {
                c_sum += A[row * N + k] * B[k * N + thread_id];
            }
            C[row * N + thread_id] = c_sum;
        }
    }
}

void matrix_mult(float *A, float *B, float *C, int N, bool flag) {
    int size = N * N * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: Allocate device memory for A, B and C
    // copy A and B to device memory. 
    cudaError_t err = cudaMalloc((void**)&A_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&B_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    err = cudaMalloc((void**)&C_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
    
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads. 
    // to perform the matrix multiplication. 
    dim3 dimGrid((N + 32 - 1) / 32.0, 1, 1);
    dim3 dimBlock(32, 1, 1);
    if (flag == true){
        row_matrix_mult_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);
    }
    else {
        col_matrix_mult_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);
    }
    
    // Part 3: Copy the result back to the host. 
    // free the device memory. 
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 5;
    
    int size = N * N * sizeof(float);

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    // Initialize the matrices. 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int offset = i * N + j;
            A[offset] = rand() % 6;
            B[offset] = rand() % 6;
        }
    }

    // Print the matrices. 
    printf("\nMatrix A\n");
    print_matrix(A, N, N);
    printf("\nMatrix B\n");
    print_matrix(B, N, N);

    // Row matrix multiplication
    matrix_mult(A, B, C, N, true); 
    printf("\nMatrix C with row thread\n");
    print_matrix(C, N, N);

    // Column matrix multiplication
    matrix_mult(A, B, C, N, false); 
    printf("\nMatrix C with col thread\n");
    print_matrix(C, N, N);

    free(A);
    free(B);
    free(C);
    
    return 0;
}
