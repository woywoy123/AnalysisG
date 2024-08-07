set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/Experimental.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/Experimental.cxx)

add_library(experimental STATIC ${SOURCE_FILES})
target_include_directories(experimental PRIVATE include/models)
target_include_directories(experimental PUBLIC include)
target_link_libraries(experimental LINK_PUBLIC cmodel nusol-cuda graph-cuda)
target_compile_options(experimental PRIVATE -fPIC)

add_custom_command(
OUTPUT Experimental.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/Experimental.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/Experimental.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/Experimental.pyx 
    --output-file Experimental.cpp
)

python_add_library(Experimental MODULE Experimental.cpp WITH_SOABI)
target_link_libraries(Experimental LINK_PUBLIC experimental)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/Experimental.cpp)

install(TARGETS Experimental DESTINATION ${SKBUILD_PROJECT_NAME}/model)
file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

