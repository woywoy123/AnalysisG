cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX C)
include(FetchContent)

#include(CheckLanguage)
#check_language(CUDA)
#
#if (CMAKE_CUDA_COMPILER)
#enable_language(CUDA)
#set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#message(STATUS "INSTALLING EXTENSIONS WITH CUDA!")
#string(REGEX REPLACE ".[0-9][0-9]|\\." "" CUDA_V ${CMAKE_CUDA_COMPILER_VERSION})
#message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
#message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
#message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CUDA_V}")
#endif()
#
#if (DEFINED ENV{ATLAS_LOCAL_ROOT_BASE})
#    message("Setup up ATLAS with 'lsetup hdf5 1.8.17-sl7'!")
#endif()
#
#if (DEFINED ENV{ALRB_HDF5_PATH})
#    set(ENV{PATH} $ENV{ALRB_HDF5_PATH}:$ENV{PATH})
#endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(HDF5 COMPONENTS C CXX REQUIRED)
find_package(ROOT COMPONENTS RIO Tree REQUIRED)
#find_package(Torch REQUIRED)

set(HDF5_USE_CXX TRUE)
include(${ROOT_USE_FILE})
include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
link_directories(${ROOT_LIBRARY_DIR})
message(STATUS "ROOT: ${ROOT_INCLUDE_DIRS}")


find_package(RapidJSON)
if (NOT RapidJSON_FOUND)
FetchContent_Declare(RapidJSON GIT_REPOSITORY https://github.com/Tencent/rapidjson.git)
FetchContent_MakeAvailable(RapidJSON)
find_package(RapidJSON REQUIRED)
endif()

add_subdirectory(src/analysisg/modules)

add_subdirectory(src/analysisg/core)
add_subdirectory(src/analysisg/events)
add_subdirectory(src/analysisg/generators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/AnalysisG/__init__.py DESTINATION AnalysisG)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/AnalysisG/IO/__init__.py DESTINATION AnalysisG/IO)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/AnalysisG/Templates/__init__.py DESTINATION AnalysisG/Templates)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/AnalysisG/Events/__init__.py DESTINATION AnalysisG/Events)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/AnalysisG/Generators/__init__.py DESTINATION AnalysisG/Generators)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/AnalysisG/Notification/__init__.py DESTINATION AnalysisG/Notification)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/analysisg/__init__.pxd DESTINATION AnalysisG/analysisg)





