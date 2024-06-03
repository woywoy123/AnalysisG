add_subdirectory(structs)
add_subdirectory(tools)
add_subdirectory(particles)
add_subdirectory(event)
add_subdirectory(io)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cmodules.cxx
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/cmodules.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/cmodules.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython
        "${CMAKE_CURRENT_SOURCE_DIR}/cmodules.pyx" --output-file "${CMAKE_CURRENT_SOURCE_DIR}/cmodules.cxx"
)

python_add_library(cmodules MODULE "${CMAKE_CURRENT_SOURCE_DIR}/cmodules.cxx" WITH_SOABI)
install(TARGETS cmodules DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules)
