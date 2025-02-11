#include <stdio.h>
#include <stdlib.h>

int main() {
    // Number of available CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Number of CUDA devices = %d", devCount);

    // query properties of CUDA devices
    cudaDeviceProp devProp;
    for (unsigned int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        printf("\nMax number of threads allowed in a block = %d", devProp.maxThreadsPerBlock);
        printf("\nNumber of SM in device = %d", devProp.multiProcessorCount);
        printf("\nClock frequency of device = %d", devProp.clockRate);
        printf("\nMax threads allowed along dimenxion x = %d", devProp.maxThreadsDim[0]);
        printf("\nMax threads allowed along dimenxion y = %d", devProp.maxThreadsDim[1]);
        printf("\nMax threads allowed along dimenxion z = %d", devProp.maxThreadsDim[2]);
        printf("\nMax blocks allowed along dimension x of grid = %d", devProp.maxGridSize[0]);
        printf("\nMax blocks allowed along dimension y of grid = %d", devProp.maxGridSize[1]);
        printf("\nMax blocks allowed along dimension z of grid = %d", devProp.maxGridSize[2]);

        // Note: following field name is misleading
        printf("\nNumber of registers available in each SM = %d", devProp.regsPerBlock);
        // For some the max number of registers that a block can use is less than the total that are available in SM

        printf("\nSize of warp = %d", devProp.warpSize);
    }

    return 0;
}