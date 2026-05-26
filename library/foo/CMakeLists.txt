if(WIN32)
    find_package(CUDAToolkit REQUIRED)

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11.6")
        message(FATAL_ERROR "CUDA version must be at least 11.6")
    endif()

    set(CUDA_FLAGS_STR
        "-std=c++17 -O3 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -gencode arch=compute_80,code=sm_80"
    )

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "11.8")
        string(APPEND CUDA_FLAGS_STR " -gencode arch=compute_90,code=sm_90")
    endif()
endif()

add_library(foo SHARED)
add_library(Foo::foo ALIAS foo)

target_sources(foo PRIVATE
    src/foo.cpp
    src/foo.cu
    src/kernel.h
)

target_include_directories(foo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(WIN32)
    target_compile_definitions(foo PRIVATE CXX_BUILD)
    target_link_libraries(foo PUBLIC CUDA::cudart_static)
    target_compile_options(foo PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:${CUDA_FLAGS_STR}>")
endif()
