set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/tools.h)
set(SOURCE_FILES
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/tools.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/strings.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/io.cxx
)

add_library(ctools STATIC ${SOURCE_FILES})
target_include_directories(ctools PRIVATE include/tools)
target_include_directories(ctools PUBLIC include)
target_compile_options(ctools PRIVATE -fPIC)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/ctools.cxx
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/tools.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/tools.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        "${CMAKE_CURRENT_SOURCE_DIR}/tools.pyx" --output-file "${CMAKE_CURRENT_SOURCE_DIR}/ctools.cxx"
)

python_add_library(tools MODULE "${CMAKE_CURRENT_SOURCE_DIR}/ctools.cxx" WITH_SOABI)
target_link_libraries(tools LINK_PUBLIC ctools)
install(TARGETS tools DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules)
install(FILES ${SOURCE_FILES} DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules/tools)
install(FILES ${HEADER_FILES} DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/cmodules/tools)


