set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/metrics/metrics.h)
set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/metrics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/loss_plots.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/mass_plots.cxx
)

add_library(cmetrics STATIC ${SOURCE_FILES})
target_include_directories(cmetrics PRIVATE include/metrics)
target_include_directories(cmetrics PUBLIC include ${ROOT_INCLUDES_DIR})

if(CMAKE_ANALYSISG_CUDA)
target_link_libraries(cmetrics LINK_PUBLIC cmodel_template cupyc ${ROOT_LIBRARIES})
target_compile_definitions(cmetrics PRIVATE PYC_CUDA ON)
else()
target_link_libraries(cmetrics LINK_PUBLIC cmodel_template tpyc ${ROOT_LIBRARIES})
endif()
target_compile_options(cmetrics PRIVATE -fPIC)
