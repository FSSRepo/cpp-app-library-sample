#include <stdio.h>
#ifdef _WIN32
#include "kernel.h"
#include <Windows.h>
#define GET_PROCESS_ADDRESS GetProcAddress
#define FREE_LIBRARY FreeLibrary
#else
#include <dlfcn.h>
#define GET_PROCESS_ADDRESS dlsym
#define FREE_LIBRARY dlclose
#endif
#include "foo_api.h"

typedef void (*super_func)(float*, int);
#ifdef _WIN32
void launch_external(float* a, float* b, float* c, int n, cudaStream_t stream)
#else
void launch_external(float* a, float* b, float* c, int n)
#endif
{
    printf("Foo Library\n");
#ifdef _WIN32
    MessageBox(NULL, "libreria cuda", "Sumando con cuda", MB_OK);
    HINSTANCE s_lib = LoadLibrary(TEXT("super.dll"));
#else
    void* s_lib = dlopen("./libsuper.so", RTLD_LAZY);
#endif
    if(s_lib != NULL) {
        super_func super_print = (super_func)GET_PROCESS_ADDRESS(s_lib, "super_func");
#ifdef _WIN32
        float* test = (float*)malloc(n * sizeof(float));
        getFromDevice(a, test, n, stream);
#endif
        if(!super_print) {
            printf("Error getting address to function\n");
        } else {
#ifdef _WIN32
            super_print(test, n);
#else
            super_print(a, n);
#endif
        }
        FREE_LIBRARY(s_lib);
    } else {
        printf("super.dll not detected - skipping\n");
    }
#ifdef _WIN32
    executeKernel(a, b, c, n, stream);
#else
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
#endif
}