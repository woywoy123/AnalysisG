set(HEADER_FILES include/io/io.h)
set(SOURCE_FILES STATIC io.cxx hdf5.cxx)
add_library(io STATIC ${SOURCE_FILES})

target_include_directories(io PRIVATE include/io)
target_include_directories(io PUBLIC include)
target_include_directories(io PRIVATE ${HDF5_CXX_INCLUDE_DIRS}) 
target_link_libraries(io LINK_PUBLIC ${HDF5_CXX_LIBRARIES} hdf5::hdf5_cpp tools structs)
target_compile_options(io PRIVATE -fPIC)
