cmake_minimum_required(VERSION 3.18)

find_package(glfw3 REQUIRED)

if(WIN32)
    find_package(Vulkan REQUIRED)
    set(VK_FILES vkapp.cpp)
else()
    find_package(OpenGL REQUIRED)
    set(OGL_FILES oglapp.cpp glad/src/gl.c glad/src/egl.c)
    set(OGL_LIBS OpenGL::GL)
endif()

include_directories(${Vulkan_INCLUDE_DIRS} glad/include)

add_library(super SHARED super.cpp ${VK_FILES}  ${OGL_FILES})

target_link_libraries(super glfw  ${Vulkan_LIBRARIES} ${OGL_LIBS})

if(WIN32)
    add_custom_command(TARGET super POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert.spv $<TARGET_FILE_DIR:super>)
    add_custom_command(TARGET super POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag.spv $<TARGET_FILE_DIR:super>)
endif()