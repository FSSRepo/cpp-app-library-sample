function(copy_runtime_deps target)
    if(NOT WIN32)
        return()
    endif()

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "[RuntimeDeps] Copiando dependencias de runtime para ${target}..."
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${target}>
            $<TARGET_FILE_DIR:${target}>
        COMMAND_EXPAND_LISTS
    )
endfunction()
