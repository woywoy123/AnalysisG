set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/event/event.h)
set(SOURCE_FILES STATIC 
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/event.cxx
)

add_library(cevent STATIC ${SOURCE_FILES})
target_include_directories(cevent PRIVATE include/event)
target_include_directories(cevent PUBLIC include)
target_compile_options(cevent PRIVATE -fPIC)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cevent.cxx
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/event.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/event.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        "${CMAKE_CURRENT_SOURCE_DIR}/event.pyx" --output-file "${CMAKE_CURRENT_SOURCE_DIR}/cevent.cxx"
)

python_add_library(event MODULE "${CMAKE_CURRENT_SOURCE_DIR}/cevent.cxx" WITH_SOABI)
target_link_libraries(event LINK_PUBLIC cevent ctools cparticles cstructs)
install(TARGETS event DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules)
