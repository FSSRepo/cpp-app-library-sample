#include <stdio.h>
#include "super.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream>
#include <vector>
#include <exception>
#include "vkapp.h"

void super_func(float* arr, int c) {
    printf("Super Library:\n");
    for(int i = 0; i < c; i++) {
        printf("Item %d: %.3f\n", i, arr[i]);
    }
    printf("launching vulkan app\n");
	run_vulkan_app();
}