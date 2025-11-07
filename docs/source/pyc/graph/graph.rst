graph.h
=======

**File Path**: ``src/AnalysisG/pyc/graph/include/graph/graph.h``

**File Type**: C++ Header

**Lines**: 14

Description
-----------

std::map<std::string, torch::Tensor> edge_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature);
std::map<std::string, torch::Tensor> node_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature);

Dependencies
------------

**C++ Includes**:

- ``map``
- ``string``
- ``torch/torch.h``

