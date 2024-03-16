
find_package(Vulkan REQUIRED)

if(WIN32)
    find_package(GLFW3 3.3 REQUIRED)
else()
    find_package(glfw3 3.3 REQUIRED)
endif()

include_directories(${Vulkan_INCLUDE_DIRS})

add_library(super SHARED super.cpp vkapp.cpp)

target_link_libraries(super glfw ${Vulkan_LIBRARIES})

add_custom_command(TARGET super POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert.spv $<TARGET_FILE_DIR:super>)
add_custom_command(TARGET super POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag.spv $<TARGET_FILE_DIR:super>)