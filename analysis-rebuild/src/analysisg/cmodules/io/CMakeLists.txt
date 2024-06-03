set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/io/io.h)
set(SOURCE_FILES STATIC
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/io.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/hdf5.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/types.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/root.cxx
)
add_library(cio STATIC ${SOURCE_FILES})
target_include_directories(cio PRIVATE include/io)
target_include_directories(cio PUBLIC include)
target_include_directories(cio PUBLIC ${HDF5_CXX_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS}) 
target_link_libraries(cio LINK_PUBLIC ${HDF5_CXX_LIBRARIES} ${ROOT_LIBRARIES} hdf5::hdf5_cpp ctools cstructs)
target_compile_options(cio PRIVATE -fPIC)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cio.cxx
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/io.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/io.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        "${CMAKE_CURRENT_SOURCE_DIR}/io.pyx" --output-file "${CMAKE_CURRENT_SOURCE_DIR}/cio.cxx"
)

python_add_library(io MODULE "${CMAKE_CURRENT_SOURCE_DIR}/cio.cxx" WITH_SOABI)
target_link_libraries(io LINK_PUBLIC cio)
install(TARGETS io DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules)

