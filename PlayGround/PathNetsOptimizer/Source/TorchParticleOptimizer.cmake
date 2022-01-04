@PACKAGE_INIT@

set(PN TorchParticleOptimizer)
set(${PN}_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/@CMAKE_INSTALL_INCLUDEDIR@")
set(${PN}_LIBRARY "")
set(${PN}_DEFINITIONS USING_${PN})

check_required_components(${PN})

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development)
target_link_libraries(TorchParticleOptimizer::TorchParticleOptimizer INTERFACE ${TORCH_LIBRARIES} Python3::Python)
if (@WITH_CUDA@)
  target_compile_definitions(TorchParticleOptimizer::TorchParticleOptimizer INTERFACE WITH_CUDA)
endif()

