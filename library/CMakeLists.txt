cmake_minimum_required(VERSION 3.18)

project(FooLibrary)

if(WIN32)
  find_package(CUDA REQUIRED)

  if(NOT CUDA_VERSION VERSION_GREATER_EQUAL "11.6")
    message(FATAL_ERROR "CUDA version must be at least 11.6")
  endif()

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++17;-O3;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math;-gencode;arch=compute_80,code=sm_80;)

  if(CUDA_VERSION VERSION_GREATER "11.8")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_90,code=sm_90)
  endif()
endif()


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set CMAKE_CXX_FLAGS to make sure -DNDEBUG is not set
set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DCXX_BUILD " CACHE STRING "Release flags" FORCE)

# require c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(vkexample)

include_directories(
    ${CUDA_INCLUDE_DIRS}
)

if(WIN32)
  cuda_add_library(foo SHARED foo.cpp foo.cu)
else()
  add_library(foo SHARED foo.cpp)
endif()