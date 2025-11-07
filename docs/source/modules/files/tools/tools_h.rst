tools.h
=======

**File Path**: ``modules/tools/include/tools/tools.h``

**File Type**: H (Header)

**Lines**: 117

Dependencies
------------

**Includes**:

- ``cstdint``
- ``iostream``
- ``map``
- ``string``
- ``vector``

Classes
-------

``tools``
~~~~~~~~~

**Methods**:

- ``void create_path(std::string path)``
- ``void delete_path(std::string path)``
- ``bool is_file(std::string path)``
- ``void rename(std::string start, std::string target)``
- ``string absolute_path(std::string path)``
- ``vector<std::string> ls(std::string path, std::string ext = "")``
- ``string to_string(double val)``
- ``string to_string(double val, int prec)``
- ``void replace(std::string* in, std::string repl_str, std::string...)``
- ``bool has_string(std::string* inpt, std::string trg)``

