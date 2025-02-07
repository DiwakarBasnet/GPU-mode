%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BLUR_SIZE 2


void printMatrix(unsigned char *matrix, int w, int h) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++){
            printf("%d ", matrix[i * w + j]);
        }
        printf("\n");
    }
    printf("\n");
}


__global__ void blurKernel(
    unsigned char *in, unsigned char *out,
    int w, int h
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        // Get avg of surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow=-BLUR_SIZE; blurRow<BLUR_SIZE+1; blurRow++) {
            for (int blurCol=-BLUR_SIZE; blurCol<BLUR_SIZE+1; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Verify we have a valid image pixel
                if (curRow>0 && curRow<h && curCol>0 && curCol<w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;       // no. of pixels for average
                }
            }
            out[row * w + col] = (unsigned char)(pixVal / pixels);
        }
    }
}


void blur(
    unsigned char *in_h, unsigned char *out_h,
    int w, int h
) {
    int size = w * h * sizeof(unsigned char);
    unsigned char *in_d, *out_d;

    // Part 1: Allocate device memory for in_d and out_d
    // copy in_h to device
    cudaError_t err1 = cudaMalloc((void**)&in_d, size);
    if (err1 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err1), __FILE__, __LINE__);
    }
    cudaError_t err2 = cudaMalloc((void**)&out_d, size);
    if (err2 != cudaSuccess) {
        printf("%s in %s at line %d", cudaGetErrorString(err2), __FILE__, __LINE__);
    }

    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);

    // Part 2: Call the kernel to launch the grid of threads
    // to perform blurring
    dim3 dimGrid(ceil(w/4.0), ceil(h/4.0), 1);
    dim3 dimBlock(4, 4, 1);
    blurKernel<<<dimGrid,dimBlock>>>(in_d, out_d, w, h);

    // Part 3: Copy the result from device to host
    // free the memory
    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);
    
    cudaFree(in_d);
    cudaFree(out_d);
}


int main() {
    int height = 20;
    int width = 25;

    int size = height * width * sizeof(unsigned char);

    // Allocate memory for host matrices
    unsigned char *in_h = (unsigned char *)malloc(size);
    unsigned char *out_h = (unsigned char *)malloc(size);

    // Initialize matrix in_h with random values
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        in_h[i] = rand() % 256;       // [0, 256]
    }

    // Print the input matrix
    printf("Matrix in_h:\n");
    printMatrix(in_h, width, height);

    // Initialize the kernel
    blur(in_h, out_h, width, height);

    // Print the result
    printf("Matrix out_h:\n");
    printMatrix(out_h, width, height);

    // Free allocated memory
    free(in_h);
    free(out_h);

    return 0;
}
