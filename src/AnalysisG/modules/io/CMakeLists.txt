set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/io/io.h)
set(SOURCE_FILES
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/io.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/hdf5.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/types.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/root.cxx
)
add_library(cio STATIC ${SOURCE_FILES})
target_include_directories(cio PRIVATE include/io)
target_include_directories(cio PUBLIC include)
target_include_directories(cio PUBLIC ${HDF5_CXX_INCLUDE_DIRS} ${ROOT_INCLUDE_DIRS}) 
target_link_libraries(cio LINK_PUBLIC 
    ${HDF5_CXX_LIBRARIES} hdf5::hdf5_cpp 
    ctools cstructs cmeta cnotification
)
target_compile_options(cio PRIVATE -fPIC)
