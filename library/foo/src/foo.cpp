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
#include "foo/foo_api.h"

typedef void (*super_func)(float*, int);
typedef void (*test_app_print)(int, float);

#ifdef _WIN32
void launch_external(float* a, float* b, float* c, int n, cudaStream_t stream)
#else
void launch_external(float* a, float* b, float* c, int n)
#endif
{
    printf("Foo Library -- launch_external\n");
#ifdef _WIN32
    printf("Get arrays from gpu\n");
    float* a_array = (float*)malloc(n * sizeof(float));
    float* b_array = (float*)malloc(n * sizeof(float));
    getFromDevice(a, a_array, n, stream);
    getFromDevice(b, b_array, n, stream);
    printf("Loading super.dll\n");
    HINSTANCE s_lib = LoadLibrary(TEXT("super.dll"));

    printf("a array values:\n");
    uintptr_t base = (uintptr_t)GetModuleHandle(NULL);
    test_app_print tap_func = (test_app_print)(base + 0x1270);
    for(int i = 0; i < n;i ++) {
        tap_func(i, a_array[i]);
    }
    printf("\n");
#else
    printf("a array values:\n");
    void* handle = dlopen(NULL, RTLD_NOW);
    test_app_print tap_func = (test_app_print)dlsym(handle, "print_item");
    if(tap_func) {
        for(int i = 0; i < n; i++) {
            tap_func(i, a[i]);
        }
    } else {
        printf("Warning: print_item not found\n");
    }
    dlclose(handle);
    printf("\n");
    void* s_lib = dlopen("./libsuper.so", RTLD_LAZY);
#endif
#ifdef _WIN32
    executeKernel(a, b, c, n, stream);
    MessageBox(NULL, "libreria cuda", "Sumando con cuda", MB_OK);
#else
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
#endif
    if(s_lib != NULL) {
        super_func super_print = (super_func)GET_PROCESS_ADDRESS(s_lib, "super_func");
        if(!super_print) {
            printf("Error getting address to function\n");
        } else {
#ifdef _WIN32
            super_print(b_array, n);
#else
            super_print(b, n);
#endif
        }
        FREE_LIBRARY(s_lib);
    } else {
        printf("super.dll not detected - skipping\n");
    }
#ifdef _WIN32
    free(a_array);
    free(b_array);
#endif
}
