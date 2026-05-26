#include <cuda_runtime.h>
#ifndef __KERNEL__
#define __KERNEL__
void getFromDevice(float* a, float* dst, int n, cudaStream_t stream);
void executeKernel(float* a, float* b, float* c, int n, cudaStream_t stream);
#endif