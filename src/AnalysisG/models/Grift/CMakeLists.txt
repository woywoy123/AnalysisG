set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/grift.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/grift.cxx)

add_library(cgnn_grift STATIC ${SOURCE_FILES})
target_include_directories(cgnn_grift PRIVATE include/models)
target_include_directories(cgnn_grift PUBLIC include)

if (CMAKE_ANALYSISG_CUDA)
target_link_libraries(cgnn_grift LINK_PUBLIC cmodel_template cupyc)
target_compile_definitions(cgnn_grift PRIVATE PYC_CUDA ON)
else()
target_link_libraries(cgnn_grift LINK_PUBLIC cmodel_template tpyc)
endif()
target_compile_options(cgnn_grift PRIVATE -fPIC)

add_custom_command(OUTPUT GRIFT.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/Grift.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/Grift.pyx
VERBATIM COMMAND Python::Interpreter -m cython 
    --cplus ${CMAKE_CURRENT_SOURCE_DIR}/Grift.pyx 
    --output-file GRIFT.cpp
)

python_add_library(Grift    MODULE GRIFT.cpp WITH_SOABI)
target_link_libraries(Grift PUBLIC cgnn_grift)
install(TARGETS Grift       DESTINATION ${SKBUILD_PROJECT_NAME}/models)
