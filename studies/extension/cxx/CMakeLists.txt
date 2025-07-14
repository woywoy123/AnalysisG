cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(SKBUILD_PROJECT_NAME AnalysisG)
project(${SKBUILD_PROJECT_NAME}  VERSION 5.0 LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(SOURCE_FILES 
    nunu.h 
    base.h 
    particle.h 
    matrix.h
    tools.h 
    nunu.cxx 
    base.cxx 
    particle.cxx 
    matrix.cxx
    tools.cxx
)

add_library(cnunu STATIC ${SOURCE_FILES})
target_compile_options(cnunu PRIVATE -Wall -fPIC)

add_custom_command(
    OUTPUT cnunu.cpp DEPENDS nunu.pyx nunu.pxd        
    VERBATIM COMMAND Python::Interpreter -m cython 
    --capi-reexport-cincludes 
    --no-docstrings
    --verbose 
    --output-file cnunu.cpp
    --cplus nunu.pyx
)

python_add_library(nunu MODULE cnunu.cpp WITH_SOABI)
target_link_libraries(nunu PUBLIC cnunu)
target_compile_options(nunu PRIVATE -fPIC)
install(TARGETS nunu DESTINATION .. )

