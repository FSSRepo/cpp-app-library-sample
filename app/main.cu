#include <cuda_runtime.h>
#include "foo_api.h"

void run_cuda(float* a, float* b, float* c, int n) {
    float *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice, stream);

    // launch library kernel
    launch_external(d_a, d_b, d_c, n, stream);

    cudaMemcpyAsync(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}