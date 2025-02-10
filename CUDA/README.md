# CUDA
This repository contains notes and exercises from the book **Programming Massively Parallel Processors**.

## 🚀 Progress

### Day-01
Today I read chapter 1 of the book, which includes need for parallel computing and its benefits.
- [x] Difference in architecture of a CPU and a GPU
- [x] Hello world program in CPU
- [x] Hello world program in GPU

### Day-02
Today I read chapter 2 of the book, "*Heterogeneous data parallel computing*"
- [x] Function declarations (`kernel function`, `device function`, `host function`)
- [x] Built-in variables (`threadIdx`, `blockIdx`, `blockDim`)
- [x] Kerne call and grid launches (```<<<gridDim, blockDim>>>```)
- [x] CUDA vector addition

### Day-03
Today I learnt about creating kernels for operating on multidimensional arrays.
- [x] Kernel for converting color of image to grayscale
- [x] Kernel for image blur effect
- [x] Square matrix multiplication in CUDA

### Day-04
Today I continued the exercises listed in chapter 3 of the book, which is related to matirx multiplication.
- [x] Matrix-vector multiplication
- [x] Matrix multiplication where each thread calculates a row
- [x] Matrix multiplication where each thread calculates a column

### Day-05
Today I read chapter 4 of the book, "*Compute architecture and scheduling*". Which contains theories related to 
modern architectire of a GPU and warp.
- [x] synchronization of threads
= [x] Warp scheduling