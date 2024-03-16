#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32)
#include "main_cuda.h"
#else
#include "foo_api.h"
#endif

void item(int i, float x) {
    printf("Item %d = %.4f\n", i, x);
}

int main(int argc, char* args[]) {
    constexpr int count = 3;

    float a[count] = {
        1.0f, 2.0f, 3.0f
    };

    float b[count] = {
        3.0f, 2.0f, 1.0f
    };

    float* res = (float*)malloc(count * sizeof(float));
    res[0] = 3.0f;

    printf("item address = %p\n", &item);

#if defined(_WIN32)
    run_cuda(a, b, res, count);
#else
    launch_external(a, b, res, count);
#endif

    printf("NodeJS: ");
    for(int i = 0; i < count; i++) {
        printf("%.2f ", res[i]);
    }
}