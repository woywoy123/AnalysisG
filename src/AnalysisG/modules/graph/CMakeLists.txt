set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/templates/graph_template.h)
set(SOURCE_FILES 
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/graph_template.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/struct_graph.cxx
       ${CMAKE_CURRENT_SOURCE_DIR}/cxx/properties.cxx
)

add_library(cgraph_template STATIC ${SOURCE_FILES})
target_include_directories(cgraph_template PRIVATE include/templates)
target_include_directories(cgraph_template PUBLIC include)

if(CMAKE_ANALYSISG_CUDA)
target_link_libraries(cgraph_template  PUBLIC cevent_template ctypecasting cupyc ${TORCH_LIBRARIES})
target_compile_definitions(cgraph_template PRIVATE PYC_CUDA ON)
else()
target_link_libraries(cgraph_template  PUBLIC cevent_template ctypecasting tpyc ${TORCH_LIBRARIES})
endif()
target_compile_options(cgraph_template PRIVATE -fPIC)
target_compile_options(cgraph_template PRIVATE -Wall -pedantic -Wextra)
