optimizer.h
===========

**File Path**: ``modules/structs/include/structs/optimizer.h``

**File Type**: H (Header)

**Lines**: 106

Dependencies
------------

**Includes**:

- ``string``
- ``structs/enums.h``
- ``structs/property.h``
- ``vector``

Classes
-------

``optimizer_params_t``
~~~~~~~~~~~~~~~~~~~~~~

**Methods**:

- ``static set_eps(double*, optimizer_params_t* obj)``
- ``static set_lr(double*, optimizer_params_t* obj)``
- ``static set_lr_decay(double*, optimizer_params_t* obj)``
- ``static set_weight_decay(double*, optimizer_params_t* obj)``
- ``static set_initial_accumulator_value(double*, optimizer_params_t* obj)``
- ``static set_beta_hack(std::vector<float>* val, optimizer_params_t* obj)``
- ``static set_betas(std::tuple<float, float>*, optimizer_params_t* obj)``
- ``static set_amsgrad(bool*, optimizer_params_t* obj)``
- ``static set_max_iter(int*, optimizer_params_t* obj)``
- ``static set_max_eval(int*, optimizer_params_t* obj)``

Structs
-------

``loss_opt``
~~~~~~~~~~~~

**Members**:

- ``loss_enum fx = loss_enum::invalid_loss``
- ``bool mean = false``
- ``bool sum  = false``
- ``bool none = false``
- ``bool swap = false``
- ``bool full = false``
- ``bool batch_mean = false``
- ``bool target     = false``
- ``bool zero_inf   = false``
- ``bool defaults   = true``

