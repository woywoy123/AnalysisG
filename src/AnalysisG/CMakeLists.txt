set(ANALYSISG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "ANALYSIG-SOURCE-DIR: ${ANALYSISG_SOURCE_DIR}")

function(cybuild path out name links)
file(REMOVE c${name}.cpp)
add_custom_command(OUTPUT c${name}.cpp
DEPENDS 
    ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pxd
    ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx
VERBATIM COMMAND Python::Interpreter -m cython 
    --cplus ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx 
    --output-file c${name}.cpp
)
python_add_library(${name} MODULE c${name}.cpp WITH_SOABI)
target_link_libraries(${name} PUBLIC c${name} "${links}")
target_compile_options(${name} PRIVATE -fPIC)
install(TARGETS ${name} DESTINATION ${out})
endfunction()


add_subdirectory(pyc)
add_subdirectory(modules)
add_subdirectory(generators)
add_subdirectory(events)
add_subdirectory(graphs)
add_subdirectory(selections)
add_subdirectory(models)
add_subdirectory(core)


