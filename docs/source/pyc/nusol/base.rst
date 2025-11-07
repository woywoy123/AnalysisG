base.cuh
========

**File Path**: ``src/AnalysisG/pyc/nusol/include/nusol/base.cuh``

**File Type**: CUDA Header

**Lines**: 42

Description
-----------

std::map<std::string, torch::Tensor> BaseDebug(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);
std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses);
std::map<std::string, torch::Tensor> BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, double mT, double mW, double mN);
std::map<std::string, torch::Tensor> Intersection(torch::Tensor* A, torch::Tensor* B, double nulls);
std::map<std::string, torch::Tensor> Nu(torch::Tensor* H, torch::Tensor* sigma, torch::Tensor* met_xy, double null);

Dependencies
------------

**C++ Includes**:

- ``map``
- ``string``
- ``torch/torch.h``
- ``utils/atomic.cuh``

