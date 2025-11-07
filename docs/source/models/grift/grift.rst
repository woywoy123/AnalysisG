grift.h
=======

**File Path**: ``src/AnalysisG/models/grift/include/models/grift.h``

**File Type**: C++ Header

**Lines**: 51

Description
-----------

model_template* clone() override;
void forward(graph_t*) override;
torch::Tensor node_encode(torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn);

Dependencies
------------

**C++ Includes**:

- ``templates/model_template.h``

Classes
-------

``grift``
~~~~~~~~~

Class defined in this file.

