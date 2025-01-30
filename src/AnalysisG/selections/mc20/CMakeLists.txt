add_subdirectory(topkinematics)
add_subdirectory(topmatching)
add_subdirectory(zprime)

file(INSTALL __init__.py  DESTINATION . )
file(INSTALL __init__.pxd DESTINATION . )
