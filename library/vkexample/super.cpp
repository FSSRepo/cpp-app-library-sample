#include <stdio.h>
#include "super.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream>
#include <vector>
#include <exception>
#if defined(_WIN32) && !defined(OGL_APP)
#include "vkapp.h"
#else
#include "oglapp.h"
#endif

void super_func(float* arr, int c) {
    printf("Super Library:\n");
    for(int i = 0; i < c; i++) {
        printf("Item %d: %.3f\n", i, arr[i]);
    }
    printf("launching graphic app\n");
#if defined(_WIN32) && !defined(OGL_APP)
	run_vulkan_app();
#else
    run_opengl_app();
#endif
}