set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork/RecursiveGraphNeuralNetwork.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork/RecursiveGraphNeuralNetwork.cxx)

add_library(recursivegraphneuralnetwork STATIC ${SOURCE_FILES})
target_include_directories(recursivegraphneuralnetwork PUBLIC RecursiveGraphNeuralNetwork)
target_link_libraries(recursivegraphneuralnetwork 
    LINK_PUBLIC cmodel nusol-cuda graph-cuda
)

target_compile_options(recursivegraphneuralnetwork PRIVATE -fPIC)

add_custom_command(
OUTPUT RecursiveGraphNeuralNetwork.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork/__init__.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork/RecursiveGraphNeuralNetwork.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork/RecursiveGraphNeuralNetwork.pyx
VERBATIM
COMMAND
    Python::Interpreter -m cython --cplus 
    ${CMAKE_CURRENT_SOURCE_DIR}/RecursiveGraphNeuralNetwork/RecursiveGraphNeuralNetwork.pyx 
    --output-file RecursiveGraphNeuralNetwork.cpp
)

python_add_library(RecursiveGraphNeuralNetwork MODULE RecursiveGraphNeuralNetwork.cpp WITH_SOABI)
target_link_libraries(RecursiveGraphNeuralNetwork LINK_PUBLIC recursivegraphneuralnetwork)

install(FILES   __init__.pxd                DESTINATION ${SKBUILD_PROJECT_NAME}/model)
install(FILES   __init__.py                 DESTINATION ${SKBUILD_PROJECT_NAME}/model)
install(TARGETS RecursiveGraphNeuralNetwork DESTINATION ${SKBUILD_PROJECT_NAME}/model)

