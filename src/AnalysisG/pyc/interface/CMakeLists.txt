if(CMAKE_ANALYSISG_CUDA)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pyc/cupyc.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/interface.cxx
)

add_library(cupyc SHARED ${SOURCE_FILES})
target_include_directories(cupyc PRIVATE include/pyc)
target_include_directories(cupyc PUBLIC include)
target_link_libraries(cupyc PUBLIC cutils cutransform cuphysics cuoperators cunusol cugraph)

set_property(TARGET cupyc PROPERTY CUDA_SEPARATE_COMPILATION ON)
target_compile_options(cupyc PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
target_compile_options(cupyc PRIVATE -fPIC)
install(TARGETS cupyc DESTINATION .)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda/transform.cxx)
add_library(transform_cuda SHARED ${SOURCE_FILES})
target_include_directories(transform_cuda PRIVATE include/pyc)
target_include_directories(transform_cuda PUBLIC include)
target_link_libraries(transform_cuda PUBLIC cupyc)
target_compile_options(transform_cuda PRIVATE -fPIC)
install(TARGETS transform_cuda DESTINATION .)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda/physics.cxx)
add_library(physics_cuda SHARED ${SOURCE_FILES})
target_include_directories(physics_cuda PRIVATE include/pyc)
target_include_directories(physics_cuda PUBLIC include)
target_link_libraries(physics_cuda PUBLIC cupyc)
target_compile_options(physics_cuda PRIVATE -fPIC)
install(TARGETS physics_cuda DESTINATION .)


set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda/operators.cxx)
add_library(operators_cuda SHARED ${SOURCE_FILES})
target_include_directories(operators_cuda PRIVATE include/pyc)
target_include_directories(operators_cuda PUBLIC include)
target_link_libraries(operators_cuda PUBLIC cupyc)
target_compile_options(operators_cuda PRIVATE -fPIC)
install(TARGETS operators_cuda DESTINATION .)


set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda/nusol.cxx)
add_library(nusol_cuda SHARED ${SOURCE_FILES})
target_include_directories(nusol_cuda PRIVATE include/pyc)
target_include_directories(nusol_cuda PUBLIC include)
target_link_libraries(nusol_cuda PUBLIC cupyc)
target_compile_options(nusol_cuda PRIVATE -fPIC)
install(TARGETS nusol_cuda DESTINATION .)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda/graph.cxx)
add_library(graph_cuda SHARED ${SOURCE_FILES})
target_include_directories(graph_cuda PRIVATE include/pyc)
target_include_directories(graph_cuda PUBLIC include)
target_link_libraries(graph_cuda PUBLIC cupyc)
target_compile_options(graph_cuda PRIVATE -fPIC)
install(TARGETS graph_cuda DESTINATION .)
endif()

set(SOURCE_FILES
    ${CMAKE_CURRENT_LIST_DIR}/include/pyc/tpyc.h
    ${CMAKE_CURRENT_LIST_DIR}/tensor/interface.cxx
)

add_library(tpyc ${SOURCE_FILES})
target_include_directories(tpyc PRIVATE include/pyc)
target_include_directories(tpyc PUBLIC include)
target_link_libraries(tpyc PUBLIC ctils ctransform cphysics cgraph cnusol coperators)
target_compile_options(tpyc PRIVATE -fPIC)
install(TARGETS tpyc DESTINATION .)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tensor/transform.cxx)
add_library(transform_tensor SHARED ${SOURCE_FILES})
target_include_directories(transform_tensor PRIVATE include/pyc)
target_include_directories(transform_tensor PUBLIC include)
target_link_libraries(transform_tensor PUBLIC tpyc)
target_compile_options(transform_tensor PRIVATE -fPIC)
install(TARGETS transform_tensor DESTINATION .)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tensor/physics.cxx)
add_library(physics_tensor SHARED ${SOURCE_FILES})
target_include_directories(physics_tensor PRIVATE include/pyc)
target_include_directories(physics_tensor PUBLIC include)
target_link_libraries(physics_tensor PUBLIC tpyc)
target_compile_options(physics_tensor PRIVATE -fPIC)
install(TARGETS physics_tensor DESTINATION .)


set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tensor/operators.cxx)
add_library(operators_tensor SHARED ${SOURCE_FILES})
target_include_directories(operators_tensor PRIVATE include/pyc)
target_include_directories(operators_tensor PUBLIC include)
target_link_libraries(operators_tensor PUBLIC tpyc)
target_compile_options(operators_tensor PRIVATE -fPIC)
install(TARGETS operators_tensor DESTINATION .)


set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tensor/nusol.cxx)
add_library(nusol_tensor SHARED ${SOURCE_FILES})
target_include_directories(nusol_tensor PRIVATE include/pyc)
target_include_directories(nusol_tensor PUBLIC include)
target_link_libraries(nusol_tensor PUBLIC tpyc)
target_compile_options(nusol_tensor PRIVATE -fPIC)
install(TARGETS nusol_tensor DESTINATION .)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tensor/graph.cxx)
add_library(graph_tensor SHARED ${SOURCE_FILES})
target_include_directories(graph_tensor PRIVATE include/pyc)
target_include_directories(graph_tensor PUBLIC include)
target_link_libraries(graph_tensor PUBLIC tpyc)
target_compile_options(graph_tensor PRIVATE -fPIC)
install(TARGETS graph_tensor DESTINATION .)


