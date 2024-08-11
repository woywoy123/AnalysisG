add_custom_command(OUTPUT canalysis.cpp
DEPENDS 
    ${ANALYSISG_SOURCE_DIR}/generators/__init__.pxd
    ${ANALYSISG_SOURCE_DIR}/generators/analysis.pxd
    ${ANALYSISG_SOURCE_DIR}/generators/analysis.pyx
VERBATIM COMMAND Python::Interpreter -m cython 
    --cplus ${ANALYSISG_SOURCE_DIR}/generators/analysis.pyx 
    --output-file canalysis.cpp
)

python_add_library(analysis    MODULE canalysis.cpp WITH_SOABI)
target_link_libraries(analysis PUBLIC canalysis )

install(TARGETS analysis  DESTINATION ${SKBUILD_PROJECT_NAME}/generators)
file(INSTALL __init__.pxd DESTINATION . )
file(INSTALL __init__.py  DESTINATION . )

