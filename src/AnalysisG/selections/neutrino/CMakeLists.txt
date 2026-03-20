add_subdirectory(validation)
add_subdirectory(combinatorial)
add_subdirectory(multisol)


file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

