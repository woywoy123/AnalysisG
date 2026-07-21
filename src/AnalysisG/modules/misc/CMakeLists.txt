set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/misc/pcm.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/main.cxx)

add_library(ag_module_cmisc_pcm ${SOURCE_FILES})
target_include_directories(ag_module_cmisc_pcm PRIVATE include)
target_include_directories(ag_module_cmisc_pcm PUBLIC include)

target_include_directories(ag_module_cmisc_pcm PUBLIC ${ROOT_INCLUDE_DIRS})
target_link_libraries(ag_module_cmisc_pcm PUBLIC ${ROOT_LIBRARIES})
target_link_libraries(ag_module_cmisc_pcm PUBLIC ag_module_cstructs_enums)
target_compile_options(ag_module_cmisc_pcm PRIVATE -fPIC -Wshadow)
