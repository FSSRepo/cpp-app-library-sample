#ifndef __SUPER__
#define __SUPER__
#if defined(_WIN32)
#ifdef CXX_BUILD
#define EXPORT __declspec(dllexport) 
#else
#define EXPORT __declspec(dllimport) 
#endif
#else
#define EXPORT
#endif
extern "C" {
    EXPORT void super_func(float* a, int c);
}
#endif