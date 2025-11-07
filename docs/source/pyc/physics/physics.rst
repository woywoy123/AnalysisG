physics.cxx
===========

**File Path**: ``src/AnalysisG/pyc/physics/physics.cxx``

**File Type**: C++ Source

**Lines**: 115

Description
-----------

torch::Tensor physics_::P2(torch::Tensor* pmc){
torch::Tensor physics_::P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
std::vector<torch::Tensor> v = {*px, *py, *pz};
torch::Tensor physics_::P(torch::Tensor* pmc){

Dependencies
------------

**C++ Includes**:

- ``cmath``
- ``physics/physics.h``
- ``utils/utils.h``

