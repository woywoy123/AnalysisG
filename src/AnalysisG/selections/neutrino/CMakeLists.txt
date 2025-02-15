if (CMAKE_ANALYSISG_CUDA)
add_subdirectory(combinatorial)
add_subdirectory(validation)
endif()

file(INSTALL __init__.py  DESTINATION . )
file(INSTALL __init__.pxd DESTINATION . )
