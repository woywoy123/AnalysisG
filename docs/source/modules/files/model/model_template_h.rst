model_template.h
================

**File Path**: ``modules/model/include/templates/model_template.h``

**File Type**: H (Header)

**Lines**: 188

Dependencies
------------

**Includes**:

- ``ATen/cuda/CUDAGraph.h``
- ``c10/cuda/CUDAStream.h``
- ``notification/notification.h``
- ``structs/model.h``
- ``structs/settings.h``
- ``templates/graph_template.h``
- ``templates/lossfx.h``

Classes
-------

``model_template``
~~~~~~~~~~~~~~~~~~

**Inherits from**: ``notification, 
    public tools``

**Methods**:

- ``void forward(graph_t* data)``
- ``void train_sequence(bool mode)``
- ``void check_features(graph_t*)``
- ``void set_optimizer(std::string name)``
- ``void initialize(optimizer_params_t*)``
- ``void clone_settings(model_settings_t* setd)``
- ``void import_settings(model_settings_t* setd)``
- ``void forward(graph_t* data, bool train)``
- ``void forward(std::vector<graph_t*> data, bool train)``
- ``void register_module(torch::nn::Sequential* data)``

Structs
-------

``graph_t``
~~~~~~~~~~~

``variable_t``
~~~~~~~~~~~~~~

``optimizer_params_t``
~~~~~~~~~~~~~~~~~~~~~~

``model_report``
~~~~~~~~~~~~~~~~

``graph_t``
~~~~~~~~~~~

