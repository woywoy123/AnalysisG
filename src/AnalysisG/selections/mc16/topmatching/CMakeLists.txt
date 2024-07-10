set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.cxx)

add_library(cmc16_topmatching STATIC ${SOURCE_FILES})
target_include_directories(cmc16_topmatching PUBLIC ../topmatching)
target_link_libraries(cmc16_topmatching LINK_PUBLIC cselection cevent_bsm_4tops)
target_compile_options(cmc16_topmatching PRIVATE -fPIC)

add_custom_command(
OUTPUT topmatching.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.pyx
VERBATIM
COMMAND 
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/topmatching.pyx 
    --output-file topmatching.cpp
)

python_add_library(topmatching MODULE topmatching.cpp WITH_SOABI)
target_link_libraries(topmatching LINK_PUBLIC cmc16_topmatching)
install(FILES   __init__.pxd  DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)
install(TARGETS   topmatching DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)
