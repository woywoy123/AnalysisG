set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/models/RecursiveGraphNeuralNetwork.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/RecursiveGraphNeuralNetwork.cxx)

add_library(recursivegraphneuralnetwork STATIC ${SOURCE_FILES})
target_include_directories(recursivegraphneuralnetwork PRIVATE include/models)
target_include_directories(recursivegraphneuralnetwork PUBLIC include)
target_link_libraries(recursivegraphneuralnetwork LINK_PUBLIC cmodel nusol-cuda graph-cuda)
target_compile_options(recursivegraphneuralnetwork PRIVATE -fPIC)

add_custom_command(
OUTPUT RecursiveGraphNeuralNetwork.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork.pyx 
    --output-file RecursiveGraphNeuralNetwork.cpp
)

python_add_library(RecursiveGraphNeuralNetwork MODULE RecursiveGraphNeuralNetwork.cpp WITH_SOABI)
target_link_libraries(RecursiveGraphNeuralNetwork LINK_PUBLIC recursivegraphneuralnetwork)
file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork.cpp)

install(TARGETS RecursiveGraphNeuralNetwork DESTINATION ${SKBUILD_PROJECT_NAME}/model)
file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

