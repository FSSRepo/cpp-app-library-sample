#ifndef __TEST_API__
#define __TEST_API__
#if defined(_WIN32)
#ifdef CXX_BUILD
#define EXPORT __declspec(dllexport) 
#else
#define EXPORT __declspec(dllimport) 
#endif
#else
#define EXPORT
#endif

#if defined(_WIN32)
EXPORT void launch_external(float* a, float* b, float* c, int n, cudaStream_t stream);
#else
EXPORT void launch_external(float* a, float* b, float* c, int n);
#endif

#endif