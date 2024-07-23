add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cevent_template.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/event_template.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/event_template.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/event_template.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cevent_template.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cselection_template.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/selection_template.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/selection_template.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/selection_template.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cselection_template.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cgraph_template.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_template.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_template.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_template.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cgraph_template.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cmodel_template.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/model_template.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/model_template.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/model_template.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cmodel_template.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cio.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/io.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/io.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/io.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cio.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cplotting.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/plotting.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/plotting.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/plotting.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cplotting.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cpp
DEPENDS 
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
    ${CMAKE_CURRENT_SOURCE_DIR}/notification.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/notification.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/notification.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cnotification.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cparticle_template.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_template.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_template.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/particle_template.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/cparticle_template.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cpp
DEPENDS 
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
    ${CMAKE_CURRENT_SOURCE_DIR}/tools.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/tools.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/tools.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/ctools.cpp
)

add_custom_command(
OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/clossfx.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/lossfx.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/lossfx.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus
    ${CMAKE_CURRENT_SOURCE_DIR}/lossfx.pyx --output-file ${CMAKE_CURRENT_SOURCE_DIR}/clossfx.cpp
)

python_add_library(tools              MODULE ${CMAKE_CURRENT_SOURCE_DIR}/ctools.cpp WITH_SOABI)
python_add_library(structs            MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cpp WITH_SOABI)
python_add_library(particle_template  MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cparticle_template.cpp WITH_SOABI)
python_add_library(notification       MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cnotification.cpp WITH_SOABI)
python_add_library(meta               MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cpp WITH_SOABI)
python_add_library(io                 MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cio.cpp WITH_SOABI)
python_add_library(plotting           MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cplotting.cpp WITH_SOABI)
python_add_library(event_template     MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cevent_template.cpp WITH_SOABI)
python_add_library(graph_template     MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cgraph_template.cpp WITH_SOABI)
python_add_library(selection_template MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cselection_template.cpp WITH_SOABI)
python_add_library(model_template     MODULE ${CMAKE_CURRENT_SOURCE_DIR}/cmodel_template.cpp WITH_SOABI)
python_add_library(lossfx             MODULE ${CMAKE_CURRENT_SOURCE_DIR}/clossfx.cpp WITH_SOABI)

target_link_libraries(tools              LINK_PUBLIC ctools)
target_link_libraries(structs            LINK_PUBLIC cstructs)
target_link_libraries(particle_template  LINK_PUBLIC cparticles cstructs)
target_link_libraries(notification       LINK_PUBLIC cnotification)
target_link_libraries(meta               LINK_PUBLIC cmeta)
target_link_libraries(io                 LINK_PUBLIC cio)
target_link_libraries(plotting           LINK_PUBLIC cplotting)
target_link_libraries(event_template     LINK_PUBLIC cevent)
target_link_libraries(selection_template LINK_PUBLIC cselection)
target_link_libraries(graph_template     LINK_PUBLIC cgraph cevent)
target_link_libraries(model_template     LINK_PUBLIC cmodel)
target_link_libraries(lossfx             LINK_PUBLIC clossfx)

target_compile_options(tools              PRIVATE -fPIC)
target_compile_options(structs            PRIVATE -fPIC)
target_compile_options(particle_template  PRIVATE -fPIC)
target_compile_options(notification       PRIVATE -fPIC)
target_compile_options(meta               PRIVATE -fPIC)
target_compile_options(io                 PRIVATE -fPIC)
target_compile_options(plotting           PRIVATE -fPIC)
target_compile_options(event_template     PRIVATE -fPIC)
target_compile_options(selection_template PRIVATE -fPIC)
target_compile_options(graph_template     PRIVATE -fPIC)
target_compile_options(model_template     PRIVATE -fPIC)
target_compile_options(lossfx             PRIVATE -fPIC)

file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/ctools.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cio.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cmeta.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cstructs.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cplotting.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cparticle_template.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cnotification.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cevent_template.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cselection_template.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cgraph_template.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/cmodel_template.cpp)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/clossfx.cpp)

install(TARGETS tools              DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS structs            DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS particle_template  DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS notification       DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS meta               DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS io                 DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS plotting           DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS event_template     DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS selection_template DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS graph_template     DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS model_template     DESTINATION ${SKBUILD_PROJECT_NAME}/core)
install(TARGETS lossfx             DESTINATION ${SKBUILD_PROJECT_NAME}/core)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )
