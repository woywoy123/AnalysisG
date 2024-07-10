set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/childrenkinematics.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/childrenkinematics.cxx)

add_library(cchildrenkinematics STATIC ${SOURCE_FILES})
target_include_directories(cchildrenkinematics PUBLIC ../childrenkinematics)
target_link_libraries(cchildrenkinematics LINK_PUBLIC cselection cevent_bsm_4tops)
target_compile_options(cchildrenkinematics PRIVATE -fPIC)

add_custom_command(
OUTPUT childrenkinematics.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/childrenkinematics.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/childrenkinematics.pyx
VERBATIM
COMMAND 
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/childrenkinematics.pyx 
    --output-file childrenkinematics.cpp
)

python_add_library(childrenkinematics MODULE childrenkinematics.cpp WITH_SOABI)
target_link_libraries(childrenkinematics LINK_PUBLIC cchildrenkinematics)
install(FILES        __init__.pxd  DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)
install(TARGETS childrenkinematics DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)


