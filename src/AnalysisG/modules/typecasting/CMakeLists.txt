set(dict_path "${PYTHON_SITE_PACKAGES}/AnalysisG")
configure_file(cfg.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/cfg.h)

set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/tensor_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/merge_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/vector_cast.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/cfg.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/typecasting.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/root.cxx
)
add_library(ctypecasting ${SOURCE_FILES})
target_include_directories(ctypecasting PRIVATE include/tools)
target_include_directories(ctypecasting PUBLIC include)
target_link_libraries(ctypecasting PUBLIC ${TORCH_LIBRARIES} ${ROOT_LIBRARIES})
target_compile_options(ctypecasting PRIVATE -fPIC)
target_compile_options(ctypecasting PRIVATE -Wall -pedantic -Wextra)
