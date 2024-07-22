set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/decaymodes.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/decaymodes.cxx)

add_library(cdecaymodes STATIC ${SOURCE_FILES})
target_include_directories(cdecaymodes PUBLIC ../decaymodes)
target_link_libraries(cdecaymodes LINK_PUBLIC cselection cevent_bsm_4tops)
target_compile_options(cdecaymodes PRIVATE -fPIC)

add_custom_command(
OUTPUT decaymodes.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/decaymodes.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/decaymodes.pyx
VERBATIM
COMMAND 
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/decaymodes.pyx 
    --output-file decaymodes.cpp
)

python_add_library(decaymodes MODULE decaymodes.cpp WITH_SOABI)
target_link_libraries(decaymodes LINK_PUBLIC cdecaymodes)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/decaymodes.cpp)
install(FILES  __init__.py  DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)
install(FILES  __init__.pxd DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)
install(TARGETS  decaymodes DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)


