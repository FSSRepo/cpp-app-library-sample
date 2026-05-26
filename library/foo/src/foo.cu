#include "kernel.h"

#define BLOCK_SIZE 256

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index >= n) {
        return;
    }
    c[index] = a[index] + b[index];
}

void getFromDevice(float* a, float* dst, int n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, a, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

void executeKernel(float* a, float* b, float* c, int n, cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(a, b, c, n);
}