#define TILE_WIDTH 16
__global__ void matrixMulKernel(float *M, float *N, float *P, int Width){

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];   // shared mem arrays
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];   // scope of sma is blocks so 1 version of Mds and Nds will be created for each block

    int bx = blockIdx.x; int by = blockIdx.y;   // 1 version of bx, by, tx and ty will be created for each thread and will reside in registers that are accessible by the thread
    int tx = threadIdx.x; int ty = threadIdx.y; // Once thread ends, the values of these variables cease to exist

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Loop over the M and N tiles required to compute P element
    float Pvalue = 0;   // Strip-mining (break long loops into phases)
    for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph) {     // Each iteration corresponds to one phase of calculation

        // Collaborative loading of M and N tiles into shared memory
        if ((Row < Width) && (ph*TILE_WIDTH + tx) < Width) {
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];    // each phase uses one tile of M and one tile of N elements
        }
        else Mds[ty][tx] = 0.0f;
        if ((ph*TILE_WIDTH + ty) < Width && Col < Width) {
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        }
        else Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    if (Row < Width) && (Col < Width) {
        P[Row * Width + Col] = Pvalue;
    }

}