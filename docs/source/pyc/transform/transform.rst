transform.cxx
=============

**File Path**: ``src/AnalysisG/pyc/transform/transform.cxx``

**File Type**: C++ Source

**Lines**: 103

Description
-----------

torch::Tensor transform_::Px(torch::Tensor* pt, torch::Tensor* phi){
return pt -> view({-1, 1}) * torch::cos(phi -> view({-1, 1}));
torch::Tensor transform_::Py(torch::Tensor* pt, torch::Tensor* phi){
return pt -> view({-1, 1}) * torch::sin(phi -> view({-1, 1}));
torch::Tensor transform_::Pz(torch::Tensor* pt, torch::Tensor* eta){

Dependencies
------------

**C++ Includes**:

- ``transform/transform.h``
- ``utils/utils.h``

