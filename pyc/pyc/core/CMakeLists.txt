function(cybuild name links)
file(REMOVE c${name}.cpp)
add_custom_command(OUTPUT c${name}.cpp
DEPENDS 
    ${CMAKE_CURRENT_SOURCE_DIR}/${name}.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/${name}.pyx
VERBATIM COMMAND Python::Interpreter -m cython 
    --cplus ${CMAKE_CURRENT_SOURCE_DIR}/${name}.pyx 
    --output-file c${name}.cpp
)
python_add_library(${name} MODULE c${name}.cpp WITH_SOABI)
target_link_libraries(${name} PUBLIC c${name} "${links}")
target_link_libraries(${name} PRIVATE ctools)
target_compile_options(${name} PRIVATE -fPIC)
install(TARGETS ${name} DESTINATION . )
endfunction()


cmake_language(CALL cybuild tools "")
cmake_language(CALL cybuild notification "")
cmake_language(CALL cybuild plotting "")

