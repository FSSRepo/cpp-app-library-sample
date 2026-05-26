## C++, Vulkan, CUDA - Ejemplo de Dependencias de Librerias en Windows

Este proyecto demuestra como se organizan y resuelven las dependencias entre librerias en Windows usando CMake moderno. Muestra una cadena de dependencias de tres niveles:

```
app (test-app.exe)
  |---> foo.dll        (link estatico + runtime)
  |       |---> super.dll  (carga dinamica con LoadLibrary)
  |               |---> glfw3.dll / vulkan-1.dll
  |---> CUDA runtime
```

### Estructura del Proyecto

```
cpp-app-library-sample/
├── CMakeLists.txt                 # Raiz: unifica salidas de binarios
├── cmake/
│   └── runtime_deps.cmake         # Helper para copiar DLLs automaticamente
├── app/
│   ├── CMakeLists.txt
│   └── src/
│       ├── main.cpp               # Entrada principal
│       ├── main.cu                # Wrapper CUDA
│       └── main_cuda.h
└── library/
    ├── CMakeLists.txt
    ├── foo/                       # Libreria compartida (CUDA)
    │   ├── CMakeLists.txt
    │   ├── include/
    │   │   └── foo/
    │   │       └── foo_api.h      # API publica
    │   └── src/
    │       ├── foo.cpp
    │       ├── foo.cu
    │       └── kernel.h           # Privado (implementacion interna)
    └── graphics/                  # Antes vkexample
        ├── CMakeLists.txt
        ├── include/
        │   └── graphics/
        │       └── super.h        # API publica
        ├── src/
        │   ├── super.cpp
        │   ├── vkapp.cpp / oglapp.cpp
        │   └── glad/              # OpenGL loader (privado)
        └── shaders/
```

### Mejoras en la Estructura

1. **CMakeLists.txt raiz**: Permite compilar todo el proyecto de una sola vez, asegurando que las dependencias se resuelvan en el orden correcto.

2. **Directorios de salida unificados**: Todos los binarios (`.exe` y `.dll`) se generan en el mismo directorio (`build/bin`). Esto elimina la necesidad de copiar manualmente DLLs para que el ejecutable las encuentre en runtime.

3. **Headers publicos vs privados**:
   - Los headers de API publica estan en `include/<nombre-lib>/`
   - Los headers internos estan en `src/`
   - Se usa `target_include_directories` con `PUBLIC` / `PRIVATE` para propagar solo los headers necesarios a cada consumidor.

4. **Aliases de targets**: `Foo::foo` y `Graphics::super` permiten referirse a las librerias de forma semantica en `target_link_libraries`.

5. **Copia automatica de DLLs**: Usando `$<TARGET_RUNTIME_DLLS:>` (CMake >= 3.21) se copian automaticamente las dependencias de terceros (como GLFW o Vulkan) al directorio del ejecutable tras la compilacion.

6. **Carga dinamica como ejemplo**: `foo.dll` carga `super.dll` en runtime mediante `LoadLibrary` / `dlopen`, demostrando como una libreria puede depender de otra sin necesidad de linkarla estaticamente.

### Requisitos

- Windows (para CUDA y Vulkan)
- CMake >= 3.21
- CUDA Toolkit >= 11.6
- GLFW 3.3
- Vulkan SDK

### Compilacion

Desde la raiz del proyecto:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Todos los binarios estaran en `build/bin/Release/`.

### Diagrama de Dependencias

```
+---------------+      target_link_libraries      +---------------+
|  test-app.exe |  ----------------------------->  |   foo.dll     |
|   (app/src)   |                                 |  (library/foo)|
+---------------+                                 +---------------+
        |                                                  |
        | CUDA::cudart_static                              | LoadLibrary("super.dll")
        v                                                  v
+---------------+                                 +---------------+
|  cudart64_*.dll|                                |  super.dll    |
+---------------+                                 | (library/gph)|
                                                  +---------------+
                                                           |
                                                           | target_link_libraries
                                                           v
                                                  +---------------+
                                                  |  glfw3.dll    |
                                                  |  vulkan-1.dll |
                                                  +---------------+
```
