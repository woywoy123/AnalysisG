set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/notification/notification.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/notification.cxx)

add_library(cnotification STATIC ${SOURCE_FILES})
target_include_directories(cnotification PRIVATE include/notification)
target_include_directories(cnotification PUBLIC include)
target_compile_options(cnotification PRIVATE -fPIC)
target_compile_options(cnotification PRIVATE -Wall -pedantic -Wextra -Wshadow)
