set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/io/io.h)
set(SOURCE_FILES
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/hdf5.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/types.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/io.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/root.cxx
)
add_library(cio STATIC ${SOURCE_FILES})
target_include_directories(cio PRIVATE include/io)
target_include_directories(cio PUBLIC include ${HDF5_CXX_INCLUDE_DIRS})

# use if HDF5 isnt installed...
#target_link_libraries(cio PUBLIC ctools cstructs cmeta cnotification hdf5_cpp-shared ROOT::Core)
#target_include_directories(cio PUBLIC include)

target_link_libraries(cio PUBLIC ctools cstructs cmeta cnotification ROOT::Core ${HDF5_LIBRARIES})
target_compile_options(cio PRIVATE -fPIC)
target_compile_options(cio PRIVATE -Wall -pedantic -Wextra)
