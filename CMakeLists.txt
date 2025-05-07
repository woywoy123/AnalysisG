cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(SKBUILD_PROJECT_NAME AnalysisG)
project(${SKBUILD_PROJECT_NAME}  VERSION 5.0 LANGUAGES CXX)

option(BUILD_DOC "Build documentation" ON)

find_package(Doxygen)
set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
message("Doxygen build started")
add_custom_target( doc_doxygen ALL
COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating documentation with Doxygen" VERBATIM)

