cmake_minimum_required(VERSION 3.18)

project(TestApp)

if (MSVC)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  set(CUDA_FLAGS -use_fast_math)
  set(CUDA_FILES main.cu)
  set(CUDA_LIBRARIES CUDA::cudart_static)

  if(CUDA_VERSION VERSION_GREATER "11.8")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_90,code=sm_90)
  endif()

  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
endif()

if(MSVC)
  link_directories(../library/build/bin/Release)
else()
  link_directories(../library/build/bin)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ../library
)

add_executable(test-app main.cpp ${CUDA_FILES})

target_link_libraries(test-app PUBLIC ${CUDA_LIBRARIES} foo)

if (MSVC)
  add_custom_command(TARGET test-app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../library/build/bin/Release/foo.dll $<TARGET_FILE_DIR:test-app>)
else()
  add_custom_command(TARGET test-app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../library/build/bin/libfoo.so $<TARGET_FILE_DIR:test-app>)
endif()