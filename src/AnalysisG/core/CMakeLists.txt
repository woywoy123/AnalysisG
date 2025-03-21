cmake_language(CALL cybuild core core tools "")
cmake_language(CALL cybuild core core structs "")
cmake_language(CALL cybuild core core notification "")
cmake_language(CALL cybuild core core meta cnotification)
cmake_language(CALL cybuild core core io "")
cmake_language(CALL cybuild core core plotting "")


cmake_language(CALL cybuild core core particle_template cstructs)
cmake_language(CALL cybuild core core event_template cparticle_template)
cmake_language(CALL cybuild core core selection_template cevent_template)
cmake_language(CALL cybuild core core graph_template cevent_template)
cmake_language(CALL cybuild core core model_template cgraph_template)
cmake_language(CALL cybuild core core lossfx "")

cmake_language(CALL cybuild core core analysis canalysis)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )
