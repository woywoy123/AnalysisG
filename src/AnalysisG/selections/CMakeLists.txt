add_subdirectory(example)
add_subdirectory(mc16)
add_subdirectory(mc20)
add_subdirectory(performance)
add_subdirectory(neutrino)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

