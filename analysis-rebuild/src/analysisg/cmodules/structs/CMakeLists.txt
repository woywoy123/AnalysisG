set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/particles.h
)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/structs.cxx)
add_library(cstructs STATIC ${SOURCE_FILES})
target_include_directories(cstructs PRIVATE include/structs)
target_include_directories(cstructs PUBLIC include)
target_compile_options(cstructs PRIVATE -fPIC)

add_custom_command(
OUTPUT 
    ${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cxx
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/structs.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/structs.pyx
VERBATIM
COMMAND
Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/structs.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cxx
)

python_add_library(structs MODULE "${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cxx" WITH_SOABI)
target_link_libraries(structs PUBLIC cstructs)
install(TARGETS structs DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules)
install(FILES ${SOURCE_FILES} DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules/structs)
install(FILES ${HEADER_FILES} DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules/structs)


