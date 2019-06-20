#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

#define BLOCK_SIZE 1024
#define TILE_SIZE BLOCK_SIZE*2
// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(unsigned int *g_data, unsigned int *sum_data, int n)
{
    __shared__ unsigned int partialSums[TILE_SIZE];
    // calculate the starting index of current block
    unsigned int blockStart = TILE_SIZE*blockIdx.x;
    // check bounds and assign zero if exceeding
    if(blockStart+threadIdx.x < n)
        partialSums[threadIdx.x] = g_data[blockStart+threadIdx.x];
    else
        partialSums[threadIdx.x] = 0;
    if(blockStart+blockDim.x+threadIdx.x < n)
        partialSums[threadIdx.x + blockDim.x] = g_data[blockStart+blockDim.x+threadIdx.x];
    else
        partialSums[threadIdx.x+blockDim.x] = 0;
    // start summing up array elements with a decreasing stride
    for(unsigned int stride = blockDim.x; stride >= 1; stride >>=1){
        __syncthreads();
        if(threadIdx.x<stride)
            partialSums[threadIdx.x] += partialSums[threadIdx.x+stride];
    }
    // store the result back into the global sum.
    if(threadIdx.x==0)
        atomicAdd(&sum_data[0] ,partialSums[0]);
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
