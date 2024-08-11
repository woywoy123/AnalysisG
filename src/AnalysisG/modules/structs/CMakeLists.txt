set(SOURCE_FILES STATIC ${CMAKE_CURRENT_SOURCE_DIR}/cxx/structs.cxx)
set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/element.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/settings.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/particles.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/property.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/event.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/meta.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/report.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/folds.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/structs/model.h
)

add_library(cstructs ${SOURCE_FILES})
target_include_directories(cstructs PRIVATE include/structs)
target_include_directories(cstructs PUBLIC include ${ROOT_INCLUDE_DIRS})
target_link_libraries(cstructs PUBLIC ${ROOT_LIBRARIES})
target_compile_options(cstructs PRIVATE -fPIC)
