add_subdirectory(combinatorial)
add_subdirectory(validation)

file(INSTALL __init__.py  DESTINATION . )
file(INSTALL __init__.pxd DESTINATION . )
