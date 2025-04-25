add_subdirectory(validation)
add_subdirectory(combinatorial)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

