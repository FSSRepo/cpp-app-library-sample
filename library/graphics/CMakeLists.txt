find_package(glfw3 REQUIRED)

if(WIN32)
    find_package(Vulkan REQUIRED)
    set(GRAPHIC_SOURCES
        src/vkapp.cpp
        src/vkapp.h
    )
    set(GRAPHIC_LIBS Vulkan::Vulkan)
else()
    find_package(OpenGL REQUIRED)
    set(GRAPHIC_SOURCES
        src/oglapp.cpp
        src/oglapp.h
        src/glad/src/gl.c
        src/glad/src/egl.c
    )
    set(GRAPHIC_LIBS OpenGL::GL)
endif()

add_library(super SHARED)
add_library(Graphics::super ALIAS super)

target_sources(super PRIVATE
    src/super.cpp
    ${GRAPHIC_SOURCES}
)

target_include_directories(super
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/glad/include
)

target_link_libraries(super PUBLIC glfw ${GRAPHIC_LIBS})

if(WIN32)
    target_compile_definitions(super PRIVATE CXX_BUILD)

    add_custom_command(TARGET super POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert.spv
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag.spv
            $<TARGET_FILE_DIR:super>
    )
endif()
