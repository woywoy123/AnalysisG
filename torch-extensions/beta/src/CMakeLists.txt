cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

add_subdirectory(transform)
add_subdirectory(physics)
add_subdirectory(interface)

install(TARGETS pyc_cuda   DESTINATION ${SKBUILD_PROJECT_NAME})
install(TARGETS pyc_tensor DESTINATION ${SKBUILD_PROJECT_NAME})
install(TARGETS pyc_float  DESTINATION ${SKBUILD_PROJECT_NAME})

# Configure the Python package directory 
install(DIRECTORY DESTINATION ${SKBUILD_PROJECT_NAME}/Transform)
install(FILES ${CMAKE_SOURCE_DIR}/py_interface/__init__.py            DESTINATION ${SKBUILD_PROJECT_NAME})
install(FILES ${CMAKE_SOURCE_DIR}/py_interface/Transform/__init__.py  DESTINATION ${SKBUILD_PROJECT_NAME}/Transform)
install(FILES ${CMAKE_SOURCE_DIR}/py_interface/Transform/transform.py DESTINATION ${SKBUILD_PROJECT_NAME}/Transform)

install(DIRECTORY DESTINATION ${SKBUILD_PROJECT_NAME}/Physics)
install(DIRECTORY DESTINATION ${SKBUILD_PROJECT_NAME}/Physics/Polar)
install(DIRECTORY DESTINATION ${SKBUILD_PROJECT_NAME}/Physics/Cartesian)
install(FILES ${CMAKE_SOURCE_DIR}/py_interface/Physics/__init__.py  DESTINATION ${SKBUILD_PROJECT_NAME}/Physics)
install(FILES ${CMAKE_SOURCE_DIR}/py_interface/Physics/Polar/__init__.py  DESTINATION ${SKBUILD_PROJECT_NAME}/Physics/Polar)
install(FILES ${CMAKE_SOURCE_DIR}/py_interface/Physics/Polar/polar.py  DESTINATION ${SKBUILD_PROJECT_NAME}/Physics/Polar)
install(FILES ${CMAKE_SOURCE_DIR}/py_interface/Physics/Cartesian/__init__.py  DESTINATION ${SKBUILD_PROJECT_NAME}/Physics/Cartesian)
install(FILES ${CMAKE_SOURCE_DIR}/py_interface/Physics/Cartesian/cartesian.py  DESTINATION ${SKBUILD_PROJECT_NAME}/Physics/Cartesian)

