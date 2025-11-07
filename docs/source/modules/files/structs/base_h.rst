base.h
======

**File Path**: ``modules/structs/include/structs/base.h``

**File Type**: H (Header)

**Lines**: 135

Dependencies
------------

**Includes**:

- ``string``
- ``structs/enums.h``
- ``vector``

Structs
-------

``bsc_t``
~~~~~~~~~

**Members**:

- ``public:

        bsc_t()``
- ``virtual ~bsc_t()``
- ``void flush_buffer()``
- ``std::string as_string()``
- ``std::string scan_buffer()``
- ``data_enum root_type_translate(std::string*)``
- ``bool element(std::vector<std::vector<std::vector<double>>>* el)``
- ``bool element(std::vector<std::vector<std::vector<long>>>*   el)``
- ``bool element(std::vector<std::vector<std::vector<int>>>*    el)``

