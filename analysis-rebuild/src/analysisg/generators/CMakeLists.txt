install(FILES __init__.pxd DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/generators)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/ceventgenerator.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/eventgenerator.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/eventgenerator.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/eventgenerator.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/ceventgenerator.cpp
)


python_add_library(eventgenerator MODULE ${CMAKE_CURRENT_SOURCE_DIR}/ceventgenerator.cpp WITH_SOABI)
target_link_libraries(eventgenerator LINK_PUBLIC ceventgenerator)
install(TARGETS eventgenerator DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/generators)

