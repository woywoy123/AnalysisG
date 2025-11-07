model.h
=======

**File Path**: ``modules/structs/include/structs/model.h``

**File Type**: H (Header)

**Lines**: 31

Dependencies
------------

**Includes**:

- ``map``
- ``string``
- ``structs/enums.h``
- ``vector``

Structs
-------

``model_settings_t``
~~~~~~~~~~~~~~~~~~~~

**Members**:

- ``opt_enum    e_optim``
- ``std::string s_optim``
- ``std::string weight_name``
- ``std::string tree_name``
- ``std::string model_name``
- ``std::string model_device``
- ``std::string model_checkpoint_path``
- ``bool inference_mode``
- ``bool is_mc``
- ``std::map<std::string, std::string> o_graph``

