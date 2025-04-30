# --------------------------------------------- #
function(cysub_build path out name)
file(REMOVE c${name}.cpp)
file(GLOB_RECURSE PYX ${ANALYSISG_SOURCE_DIR}/${path}/*.pyx)
file(GLOB_RECURSE PYD ${ANALYSISG_SOURCE_DIR}/${path}/*.pxd)

list(REMOVE_ITEM PYX "${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx")
list(REMOVE_ITEM PYD "${ANALYSISG_SOURCE_DIR}/${path}/${name}.pxd")

add_custom_command(OUTPUT c${name}.cpp DEPENDS
    ${PYX} ${PYD}
    ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx 
    ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pxd
VERBATIM COMMAND Python::Interpreter -m cython 
    --capi-reexport-cincludes 
    --no-docstrings
    --verbose 
    --3str 
    --output-file c${name}.cpp
    --cplus ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx
)

python_add_library(${name} MODULE c${name}.cpp WITH_SOABI)
target_link_libraries(${name} PUBLIC c${name} ${links})
target_link_libraries(${name} PRIVATE ctools)
target_compile_options(${name} PRIVATE -fPIC)
install(TARGETS ${name} DESTINATION ${out})

foreach(cyx IN LISTS PYX)
    cmake_path(GET cyx STEM name_)
    add_custom_command(OUTPUT c${name_}.cpp DEPENDS
        ${PYX} ${PYD}
        ${ANALYSISG_SOURCE_DIR}/${path}/${name_}.pyx 
        ${ANALYSISG_SOURCE_DIR}/${path}/${name_}.pxd
    VERBATIM COMMAND Python::Interpreter -m cython 
        --capi-reexport-cincludes 
        --no-docstrings
        --verbose 
        --3str 
        --output-file c${name_}.cpp
        --cplus ${PYX} ${ANALYSISG_SOURCE_DIR}/${path}/${name_}.pyx
    )

    python_add_library(${name_} MODULE c${name_}.cpp WITH_SOABI)
    target_link_libraries(${name_} PUBLIC c${name} ${links})
    target_link_libraries(${name_} PRIVATE ctools)
    target_compile_options(${name_} PRIVATE -fPIC)
    install(TARGETS ${name_} DESTINATION ${out})
endforeach()
endfunction()

# --------------------------------------------- #
function(cybuild path out name links)
file(REMOVE c${name}.cpp)
add_custom_command(OUTPUT c${name}.cpp DEPENDS 
    ${links}
    ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pxd 
    ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx 
VERBATIM COMMAND Python::Interpreter -m cython 
    --capi-reexport-cincludes
    --no-docstrings
    --verbose 
    --3str 
    --output-file c${name}.cpp
    --cplus ${ANALYSISG_SOURCE_DIR}/${path}/${name}.pyx
)
python_add_library(${name} MODULE c${name}.cpp WITH_SOABI)
target_link_libraries(${name} PUBLIC c${name} ${links})
target_link_libraries(${name} PRIVATE ctools)
target_compile_options(${name} PRIVATE -fPIC)
install(TARGETS ${name} DESTINATION ${out})
endfunction()
# --------------------------------------------- #

set(ANALYSISG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "ANALYSIG-SOURCE-DIR: ${ANALYSISG_SOURCE_DIR}")

add_subdirectory(pyc)
add_subdirectory(modules)
add_subdirectory(events)
add_subdirectory(graphs)
add_subdirectory(selections)
add_subdirectory(metrics)
add_subdirectory(models)
add_subdirectory(core)

file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )


