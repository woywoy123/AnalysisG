set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/meta/meta.h)
set(SOURCE_FILES STATIC
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/meta.cxx
)
add_library(cmeta STATIC ${SOURCE_FILES})
target_include_directories(cmeta PRIVATE include/meta)
target_include_directories(cmeta PUBLIC include)
target_link_libraries(cmeta LINK_PUBLIC cstructs)
target_compile_options(cmeta PRIVATE -fPIC)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cxx
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/meta.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/meta.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        "${CMAKE_CURRENT_SOURCE_DIR}/meta.pyx" --output-file "${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cxx"
)

python_add_library(meta MODULE "${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cxx" WITH_SOABI)
target_link_libraries(meta LINK_PUBLIC cmeta)
install(TARGETS meta DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules)

