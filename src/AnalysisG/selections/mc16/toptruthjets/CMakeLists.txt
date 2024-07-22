set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/toptruthjets.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/toptruthjets.cxx)

add_library(ctoptruthjets STATIC ${SOURCE_FILES})
target_include_directories(ctoptruthjets PUBLIC ../toptruthjets)
target_link_libraries(ctoptruthjets LINK_PUBLIC cselection cevent_bsm_4tops)
target_compile_options(ctoptruthjets PRIVATE -fPIC)

add_custom_command(
OUTPUT toptruthjets.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/toptruthjets.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/toptruthjets.pyx
VERBATIM
COMMAND 
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/toptruthjets.pyx 
    --output-file toptruthjets.cpp
)

python_add_library(toptruthjets MODULE toptruthjets.cpp WITH_SOABI)
target_link_libraries(toptruthjets LINK_PUBLIC ctoptruthjets)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/toptruthjets.cpp)
install(FILES   __init__.py  DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)
install(FILES   __init__.pxd DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)
install(TARGETS toptruthjets DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)


