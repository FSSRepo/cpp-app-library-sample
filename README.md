## C++, Vulkan, CUDA - Windows Library Dependency Example

This project demonstrates how to organize and resolve library dependencies on Windows using modern CMake. It showcases a three-level dependency chain:

```
app (test-app.exe)
  |---> foo.dll        (static link + runtime)
  |       |---> super.dll  (dynamic LoadLibrary)
  |               |---> glfw3.dll / vulkan-1.dll
  |---> CUDA runtime
```

### Project Structure

```
cpp-app-library-sample/
├── CMakeLists.txt
├── cmake/
│   └── runtime_deps.cmake
├── app/
│   ├── CMakeLists.txt
│   └── src/
│       ├── main.cpp
│       ├── main.cu
│       └── main_cuda.h
└── library/
    ├── CMakeLists.txt
    ├── foo/
    │   ├── CMakeLists.txt
    │   ├── include/foo/foo_api.h
    │   └── src/
    │       ├── foo.cpp
    │       ├── foo.cu
    │       └── kernel.h
    └── graphics/
        ├── CMakeLists.txt
        ├── include/graphics/super.h
        ├── src/
        │   ├── super.cpp
        │   ├── vkapp.cpp / oglapp.cpp
        │   └── glad/
        └── shaders/
```

Key design points:
- **Unified output directory**: all binaries go to `build/bin` so DLLs are found automatically at runtime.
- **Public vs private headers**: public APIs under `include/<lib>/`, internals under `src/`.
- **CMake aliases**: `Foo::foo` and `Graphics::super` for semantic linking.
- **Automatic DLL copying**: uses `$<TARGET_RUNTIME_DLLS:>` (CMake >= 3.21) to copy third-party dependencies.
- **Dynamic loading example**: `foo.dll` loads `super.dll` at runtime via `LoadLibrary` / `dlopen`.

### Requirements

- Windows
- CMake >= 3.21
- CUDA Toolkit >= 11.6
- GLFW 3.3
- Vulkan SDK

### Build

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

All binaries will be in `build/bin/Release/`.
