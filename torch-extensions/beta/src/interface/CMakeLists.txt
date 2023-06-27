add_library(pyc_cuda SHARED
    ${CMAKE_CURRENT_LIST_DIR}/pyc_cuda.h
    ${CMAKE_CURRENT_LIST_DIR}/pyc_cuda.cxx
)
target_link_libraries(pyc_cuda PUBLIC ${TORCH_LIBRARIES} Python::Python transform_cu)

add_library(pyc_tensor SHARED
    ${CMAKE_CURRENT_LIST_DIR}/pyc_tensors.h
    ${CMAKE_CURRENT_LIST_DIR}/pyc_tensors.cxx
)
target_link_libraries(pyc_tensor PUBLIC ${TORCH_LIBRARIES} Python::Python transform_ten)
