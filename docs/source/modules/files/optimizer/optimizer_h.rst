optimizer.h
===========

**File Path**: ``modules/optimizer/include/generators/optimizer.h``

**File Type**: H (Header)

**Lines**: 37

Dependencies
------------

**Includes**:

- ``generators/dataloader.h``
- ``metrics/metrics.h``
- ``structs/settings.h``
- ``templates/model_template.h``

Classes
-------

``optimizer``
~~~~~~~~~~~~~

**Inherits from**: ``tools,
    public notification``

**Methods**:

- ``void import_dataloader(dataloader* dl)``
- ``void import_model_sessions(std::tuple<model_template*, optimizer_params_t*>* ...)``
- ``void training_loop(int k, int epoch)``
- ``void validation_loop(int k, int epoch)``
- ``void evaluation_loop(int k, int epoch)``
- ``void launch_model(int k)``

