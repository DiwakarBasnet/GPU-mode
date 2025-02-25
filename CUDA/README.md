# CUDA
This repository contains notes and exercises from the book **Programming Massively Parallel Processors**.

## 🚀 Progress

### Day-01
Today I learned about the need for parallel computing and its benefits. I read chapter 1 of the book and learned about the difference in architecture of a CPU and a GPU.
- [x] Hello world program in CPU
- [x] Hello world program in GPU

### Day-02
Today I read chapter 2 of the book, in which I learned about function declarations in CUDA, built-in variables, memory allocation, memory copy, memory free, kernel launch, thread, block and grid hierarchy in CUDA.
- [x] CUDA vector addition

### Day-03
Today I learnt about creating kernels for operating on multidimensional arrays. I read chapter 3 of the book, which includes working on multi-dimensional arrays.
- [x] Kernel for converting color of image to grayscale
- [x] Kernel for image blur effect
- [x] Square matrix multiplication in CUDA

### Day-04
Today I continued the exercises listed in chapter 3 of the book, which is related to matirx multiplication. I learned about the working on matrices with different row and column dimensions.
- [x] Matrix-vector multiplication
- [x] Matrix multiplication where each thread calculates a row
- [x] Matrix multiplication where each thread calculates a column

### Day-05
Today I read chapter 4 of the book, which contains theories related to 
modern architectire of a GPU and warp. I learned about thread synchronization in CUDA and warp schedulings.
- [x] Querying CUDA device properties from host code

### Day-06
Today I read chapter 5 of the book, which talks in detail about the memory hierarchy in GPU. I learned about the Streaming Multiprocessors (SM) in CUDA and way to allocate data in shared memory for increased efficiency of GPU by reducing the global memory access. I also learned about tiling for improvement in matrix multiplication. In tiling matrix multiplication, it utilizes shared memory thus reducing the global memory traffic and increasing effective bandwidth. Compared to naive approach, tiled matrix multiplication improves the performance by more than 60%.
- [x] Tiled matrix multiplication
- [x] Dynamic allocation of tile width in tiled matrix multiplication

### Day-07
Today I continued with the exercises in chapter 5 of the book and way to implement tiled matrix multiplication for non-square matrices.
- [x] Tiled matrix multiplication for non-square matrices

### Day-08
Today I read chapter 6 of the book, which talks about memory coalescing and thread coarsening. I wrote a kernel that multiplies two matrices using coarsening multiple output tiles. I am also continuing the exercises in this chapter.
- [x] Matrix multiplication with coarsening multiple output tiles
- [x] Matrix multiplication with second matrix stored in column major format

### Day-09
Today I learned about convolution and the filter kernels used for convolution. I applied basic convolition in 1D array as well as 2D matrix with and without constant memory allocation. Constant memory has smaller memory size compared to global memory and the elements in it remain constant throughout the process.
- [x] Basic 1D convolution
- [x] Basic 2D convolution
- [x] Basic 2D convolution with constant memory allocation for filter matrix