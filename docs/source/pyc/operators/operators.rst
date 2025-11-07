operators.cxx
=============

**File Path**: ``src/AnalysisG/pyc/operators/operators.cxx``

**File Type**: C++ Source

**Lines**: 93

Description
-----------

torch::Tensor operators_::Dot(torch::Tensor* v1, torch::Tensor* v2){
torch::Tensor operators_::CosTheta(torch::Tensor* v1, torch::Tensor* v2){
torch::Tensor v1_2 = ((*v1)*(*v1)).sum(-1);
torch::Tensor v2_2 = ((*v2)*(*v2)).sum(-1);
torch::Tensor dot  = ((*v1)*(*v2)).sum(-1);

Dependencies
------------

**C++ Includes**:

- ``operators/operators.h``
- ``utils/utils.h``

