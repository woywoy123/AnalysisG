set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/grift.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/grift.cxx)

add_library(cgrift STATIC ${SOURCE_FILES})
target_include_directories(cgrift PRIVATE include/models)
target_include_directories(cgrift PUBLIC include)
target_link_libraries(cgrift PUBLIC cmodel_template cpyc)
if (CMAKE_ANALYSISG_CUDA)
target_compile_definitions(cgrift PRIVATE PYC_CUDA ON)
endif()

target_compile_options(cgrift PRIVATE -fPIC)
cmake_language(CALL cybuild "models/Grift" "models" grift "")
