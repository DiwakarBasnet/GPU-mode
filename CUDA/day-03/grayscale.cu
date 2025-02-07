%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CHANNELS 3


void printMatrix(unsigned char *matrix, int width, int height) {
    printf("\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%d ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}


__global__ void colortoGrayscaleConversionKernel(
    unsigned char *Pout,
    unsigned char *Pin,
    int width,
    int height
) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((col < width) && (row < height)) {
        int grayOffset = row * width + col;
        // RGB image has 3 channels
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];       // Red value
        unsigned char g = Pin[rgbOffset + 1];   // Green value
        unsigned char b = Pin[rgbOffset + 2];   // Blue value
        // Perform the rescaling
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}


void colortoGrayscaleConversion(
    unsigned char *Pout_h,
    unsigned char *Pin_h,
    int width,
    int height
) {
    int size = width * height * sizeof(unsigned char);
    unsigned char *Pout_d, *Pin_d;

    // Part 1: Allocate device memory for Pout and Pin
    // copy Pin to device memory
    cudaError_t err1 = cudaMalloc((void**)&Pout_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&Pin_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads
    // to perform color conversion
    dim3 dimGrid(ceil(width / 4.0), ceil(height / 4.0), 1);
    dim3 dimBlock(4, 4, 1);
    colortoGrayscaleConversionKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);

    // Part 3: Copy the result from device to host
    // free the memory
    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);

    cudaFree(Pout_d);
    cudaFree(Pin_d);
}


int main() {
    int height = 20;
    int width = 25;

    int size = height * width * sizeof(unsigned char);

    // Allocate memory for host matrices
    unsigned char *A_h = (unsigned char *)malloc(size);
    unsigned char *B_h = (unsigned char *)malloc(size);

    // Initialize matrix A with random values
    //srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        A_h[i] = rand() % 256;       // [0, 256]
    }

    // Print the input matrix
    printf("Matrix A:\n");
    printMatrix(A_h, width, height);

    // Initialize the kernel
    colortoGrayscaleConversion(B_h, A_h, width, height);

    // Print the result
    printf("Matrix B:\n");
    printMatrix(B_h, width, height);

    // Free allocated memory
    free(A_h);
    free(B_h);

    return 0;
}
