plotting.h
==========

**File Path**: ``modules/plotting/include/plotting/plotting.h``

**File Type**: H (Header)

**Lines**: 99

Dependencies
------------

**Includes**:

- ``map``
- ``notification/notification.h``
- ``structs/property.h``
- ``tools/tools.h``

Classes
-------

``plotting``
~~~~~~~~~~~~

**Inherits from**: ``tools, 
    public notification``

**Methods**:

- ``string build_path()``
- ``float get_max(std::string dim)``
- ``float get_min(std::string dim)``
- ``float sum_of_weights()``
- ``void build_error()``
- ``tuple<float, float> mean_stdev(std::vector<float>* data)``

