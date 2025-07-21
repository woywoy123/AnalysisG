set(SOURCE_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/solvers.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nusol.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/nunu.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/cxx/mtx.cxx
)

add_library(cnunu ${SOURCE_FILES})
target_include_directories(cnunu PRIVATE include/templates)
target_include_directories(cnunu PUBLIC include)
target_compile_options(cnunu PRIVATE -fPIC)
target_compile_options(cnunu PRIVATE -Wall -pedantic -Wextra -Wshadow)


add_executable(nunu main.cxx)
target_link_libraries(nunu PUBLIC cnunu)
install(TARGETS nunu DESTINATION .)
