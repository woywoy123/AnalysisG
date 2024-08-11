set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/Experimental.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/Experimental.cxx)

add_library(cgnn_experimental STATIC ${SOURCE_FILES})
target_include_directories(cgnn_experimental PRIVATE include/models)
target_include_directories(cgnn_experimental PUBLIC include)

target_link_libraries(cgnn_experimental LINK_PUBLIC cmodel_template nusol-cuda graph-cuda)
target_compile_options(cgnn_experimental PRIVATE -fPIC)

add_custom_command(OUTPUT Experimental.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/Experimental.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/Experimental.pyx
VERBATIM COMMAND Python::Interpreter -m cython 
    --cplus ${CMAKE_CURRENT_SOURCE_DIR}/Experimental.pyx 
    --output-file Experimental.cpp
)

python_add_library(Experimental    MODULE Experimental.cpp WITH_SOABI)
target_link_libraries(Experimental PUBLIC cgnn_experimental)
install(TARGETS Experimental       DESTINATION ${SKBUILD_PROJECT_NAME}/models)
