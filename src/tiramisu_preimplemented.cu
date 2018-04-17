#include <stdint.h>

#define BLOCK_SIZE 512
using data_type = p_float64;
using it_type = p_uint32;

template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __sy ncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __sy ncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __sy ncthreads();
    }
    if (tid < 32) {
        if (blockSize >= 64)
            sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32)
            sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16)
            sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8)
            sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4)
            sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2)
            sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

extern data_type reduce(data_type *buffer, it_type size) {}
