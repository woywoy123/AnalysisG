add_subdirectory(bsm_4tops)
#add_subdirectory(ssml_mc20)
install(FILES __init__.pxd DESTINATION ${SKBUILD_PROJECT_NAME}/events)
install(FILES __init__.py  DESTINATION ${SKBUILD_PROJECT_NAME}/events)

