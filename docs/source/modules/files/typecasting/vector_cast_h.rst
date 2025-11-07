vector_cast.h
=============

**File Path**: ``modules/typecasting/include/tools/vector_cast.h``

**File Type**: H (Header)

**Lines**: 130

Dependencies
------------

**Includes**:

- ``ATen/cuda/CUDAContext.h``
- ``TTree.h``
- ``c10/core/DeviceType.h``
- ``c10/cuda/CUDAStream.h``
- ``structs/base.h``
- ``structs/meta.h``
- ``torch/torch.h``
- ``vector``

Structs
-------

``write_t``
~~~~~~~~~~~

``variable_t``
~~~~~~~~~~~~~~

Functions
---------

``void tensor_vector(std::vector<g>* trgt, std::vector<g>* chnks, std::vector<sig...)``

``void tensor_vector(std::vector<G>* trgt, std::vector<g>* chnks, std::vector<sig...)``

``bool tensor_to_vector(torch::Tensor* data, std::vector<G>* out, std::vector<signed...)``

``void tensor_to_vector(torch::Tensor* data, std::vector<g>* out)``

