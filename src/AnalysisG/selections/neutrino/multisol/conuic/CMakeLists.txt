set(HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/conuic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/atomics.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/constants.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/conuic/data.h
)

set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/conuic.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/atomics.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/constants.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/data.cxx
)

add_library(cxnunix STATIC ${SOURCE_FILES})
target_include_directories(cxnunix PRIVATE include/conuic)
target_include_directories(cxnunix PUBLIC include)

target_link_libraries(cxnunix  PUBLIC cparticle_template ccommon)
target_compile_options(cxnunix PRIVATE -fPIC)

