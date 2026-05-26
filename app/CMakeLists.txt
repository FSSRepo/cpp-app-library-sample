include(runtime_deps)

if(WIN32)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CUDA_FILES src/main.cu)
    set(CUDA_LIBS CUDA::cudart_static)
    set(CUDA_FLAGS -use_fast_math)

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "11.8")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_90,code=sm_90)
    endif()

    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
endif()

add_executable(test-app src/main.cpp ${CUDA_FILES})

target_include_directories(test-app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test-app PRIVATE Foo::foo Graphics::super ${CUDA_LIBS})

if(WIN32)
    copy_runtime_deps(test-app)
endif()
