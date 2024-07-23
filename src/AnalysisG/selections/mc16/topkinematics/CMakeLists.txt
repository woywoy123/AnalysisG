set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.cxx)

add_library(cmc16_topkinematics STATIC ${SOURCE_FILES})
target_include_directories(cmc16_topkinematics PUBLIC ../topkinematics)
target_link_libraries(cmc16_topkinematics LINK_PUBLIC cselection cevent_bsm_4tops)
target_compile_options(cmc16_topkinematics PRIVATE -fPIC)

add_custom_command(
OUTPUT topkinematics.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.pyx
VERBATIM
COMMAND 
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.pyx 
    --output-file topkinematics.cpp)

python_add_library(topkinematics MODULE topkinematics.cpp WITH_SOABI)
target_link_libraries(topkinematics LINK_PUBLIC cmc16_topkinematics)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/topkinematics.cpp)
install(TARGETS topkinematics DESTINATION ${SKBUILD_PROJECT_NAME}/selections/mc16)
