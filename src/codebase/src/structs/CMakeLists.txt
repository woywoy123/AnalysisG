set(HEADER_FILES 
    include/structs/coins.h
    include/structs/trading.h
    include/structs/market.h
)

set(SOURCE_FILES STATIC exchanges.cxx)
add_library(structs STATIC ${SOURCE_FILES})
target_include_directories(structs PRIVATE include/structs)
target_include_directories(structs PUBLIC include)
target_compile_options(structs PRIVATE -fPIC)
