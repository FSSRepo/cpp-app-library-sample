## C++, Vulkan, CUDA example

### Build library

For windows requires CUDA.

Mandatory requirements:
    - GLFW 3.3
    - Vulkan SDK

```bash
cd library
mkdir build
cmake ..
cmake --build . --config Release
```

### Build app

For windows requires CUDA.

```bash
cd app
mkdir build
cmake ..
cmake --build . --config Release
```
