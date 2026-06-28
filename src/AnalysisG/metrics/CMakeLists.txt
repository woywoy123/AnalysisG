add_subdirectory(accuracy)
add_subdirectory(pagerank)
add_subdirectory(topefficiency)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

