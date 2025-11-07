nusol.cxx
=========

**File Path**: ``src/AnalysisG/pyc/nusol/tensor/nusol.cxx``

**File Type**: C++ Source

**Lines**: 561

Description
-----------

std::map<std::string, torch::Tensor> GetMasses(torch::Tensor* L, torch::Tensor* masses){
if (dim_i != dim_i_){_masses = torch::ones({dim_i, 3}, MakeOp(masses))*_masses[0];}
torch::Tensor _x0(torch::Tensor* pmc, torch::Tensor* _pm2, torch::Tensor* mH2, torch::Tensor* mL2){

Dependencies
------------

**C++ Includes**:

- ``nusol/nusol.h``
- ``operators/operators.h``
- ``physics/physics.h``
- ``transform/transform.h``
- ``utils/utils.h``

