set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/generators/optimizer.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/optimizer.cxx)

add_library(coptimizer STATIC ${SOURCE_FILES})
target_include_directories(coptimizer PRIVATE include/generators)
target_include_directories(coptimizer PUBLIC include)
target_link_libraries(coptimizer PUBLIC cmodel_template cdataloader cmetrics)
target_link_libraries(coptimizer PRIVATE ${TORCH_LIBRARIES})
target_compile_options(coptimizer PRIVATE -fPIC)
target_compile_options(coptimizer PRIVATE -Wall -pedantic -Wextra)

if(CMAKE_ANALYSISG_CUDA)
target_compile_definitions(coptimizer PRIVATE PYC_CUDA ON)
endif()



