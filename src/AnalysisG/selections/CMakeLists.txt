add_subdirectory(example)
add_subdirectory(mc16)
add_subdirectory(mc20)
add_subdirectory(mc20_compare)
add_subdirectory(performance)
add_subdirectory(neutrino)
add_subdirectory(analysis)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

