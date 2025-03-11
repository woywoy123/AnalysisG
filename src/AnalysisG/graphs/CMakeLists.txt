add_subdirectory(bsm_4tops)
add_subdirectory(ssml_mc20)
add_subdirectory(exp_mc20)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

