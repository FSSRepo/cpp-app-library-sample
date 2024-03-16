cmake_minimum_required(VERSION 3.18)

project(TestApp)

find_package(CUDAToolkit REQUIRED)

link_directories(../library/build/bin/Release)

enable_language(CUDA)

set(CUDA_FLAGS -use_fast_math)

if(CUDA_VERSION VERSION_GREATER "11.8")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_90,code=sm_90)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ../library
)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")

add_executable(test-app main.cpp main.cu)

target_link_libraries(test-app PUBLIC CUDA::cudart_static foo)


if (MSVC)
  add_custom_command(TARGET test-app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../library/build/bin/Release/foo.dll $<TARGET_FILE_DIR:test-app>)
else()
  add_custom_command(TARGET test-app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../library/build/bin/libfoo.so $<TARGET_FILE_DIR:test-app>)
endif()