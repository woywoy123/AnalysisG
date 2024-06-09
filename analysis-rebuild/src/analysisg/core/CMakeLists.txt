install(FILES __init__.pxd DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/core)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cevent_template.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_template.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_template.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/event_template.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cevent_template.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cio.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/io.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/io.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/io.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cio.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/meta.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/meta.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/meta.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cnotification.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/notification.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/notification.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/notification.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cnotification.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cparticles.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_template.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_template.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_template.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cparticles.cpp
)

add_custom_command(
OUTPUT 
    ${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/structs.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/structs.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/structs.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cpp
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/ctools.cpp
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/tools.pxd
        ${CMAKE_CURRENT_SOURCE_DIR}/tools.pyx

    VERBATIM
    COMMAND
        Python::Interpreter -m cython --cplus
        ${CMAKE_CURRENT_SOURCE_DIR}/tools.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/ctools.cpp
)

python_add_library(tools MODULE ${CMAKE_CURRENT_SOURCE_DIR}/ctools.cpp WITH_SOABI)
target_link_libraries(tools LINK_PUBLIC ctools)

python_add_library(structs MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cpp WITH_SOABI)
target_link_libraries(structs LINK_PUBLIC cstructs)

python_add_library(particle_template MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cparticles.cpp WITH_SOABI)
target_link_libraries(particle_template LINK_PUBLIC cparticles cstructs)

python_add_library(notification MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cnotification.cpp WITH_SOABI)
target_link_libraries(notification LINK_PUBLIC cnotification)

python_add_library(meta MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cpp WITH_SOABI)
target_link_libraries(meta LINK_PUBLIC cmeta)

python_add_library(io MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cio.cpp WITH_SOABI)
target_link_libraries(io LINK_PUBLIC cio)

python_add_library(event_template MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cevent_template.cpp WITH_SOABI)
target_link_libraries(event_template LINK_PUBLIC cevent)

install(TARGETS tools DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/core)
install(TARGETS io DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/core)
install(TARGETS meta DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/core)
install(TARGETS structs DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/core)
install(TARGETS particle_template DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/core)
install(TARGETS notification DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/core)
install(TARGETS event_template DESTINATION AnalysisG/${SKBUILD_PROJECT_NAME}/core)
