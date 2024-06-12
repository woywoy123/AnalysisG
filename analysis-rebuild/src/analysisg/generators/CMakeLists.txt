install(FILES __init__.pxd DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/generators)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/csampletracer.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/sampletracer.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/sampletracer.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/sampletracer.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/csampletracer.cpp
)

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

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cgraphgenerator.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/graphgenerator.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/graphgenerator.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/graphgenerator.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cgraphgenerator.cpp
)

python_add_library(sampletracer MODULE ${CMAKE_CURRENT_SOURCE_DIR}/csampletracer.cpp WITH_SOABI)
target_link_libraries(sampletracer LINK_PUBLIC csampletracer)

python_add_library(eventgenerator MODULE ${CMAKE_CURRENT_SOURCE_DIR}/ceventgenerator.cpp WITH_SOABI)
target_link_libraries(eventgenerator LINK_PUBLIC ceventgenerator)

python_add_library(graphgenerator MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cgraphgenerator.cpp WITH_SOABI)
target_link_libraries(graphgenerator LINK_PUBLIC cgraphgenerator)

install(TARGETS sampletracer   DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/sampletracer)
install(TARGETS eventgenerator DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/generators)
install(TARGETS graphgenerator DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/generators)

