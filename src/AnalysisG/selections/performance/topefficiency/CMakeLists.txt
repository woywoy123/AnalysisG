set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topefficiency.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/topefficiency.cxx)

add_library(ctopefficiency STATIC ${SOURCE_FILES})
target_include_directories(ctopefficiency PUBLIC ../topefficiency)
target_link_libraries(ctopefficiency LINK_PUBLIC cselection cevent_bsm_4tops)
target_compile_options(ctopefficiency PRIVATE -fPIC)

add_custom_command(
OUTPUT topefficiency.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/topefficiency.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/topefficiency.pyx
VERBATIM
COMMAND 
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/topefficiency.pyx 
    --output-file topefficiency.cpp
)

python_add_library(topefficiency MODULE topefficiency.cpp WITH_SOABI)
target_link_libraries(topefficiency LINK_PUBLIC ctopefficiency)
install(FILES   __init__.pxd  DESTINATION ${SKBUILD_PROJECT_NAME}/selections/performance)
install(TARGETS topefficiency DESTINATION ${SKBUILD_PROJECT_NAME}/selections/performance)
