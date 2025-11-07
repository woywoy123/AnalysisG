roc.h
=====

**File Path**: ``modules/roc/include/plotting/roc.h``

**File Type**: H (Header)

**Lines**: 45

Dependencies
------------

**Includes**:

- ``plotting/plotting.h``

Classes
-------

``roc``
~~~~~~~

**Inherits from**: ``plotting``

**Methods**:

- ``void build_ROC(std::string name, int kfold, 
            std::vec...)``
- ``vector<roc_t*> get_ROC()``
- ``vector<g> v(y, 0)``

Structs
-------

``roc_t``
~~~~~~~~~

**Members**:

- ``int cls = 0``
- ``int kfold = 0``
- ``std::string model = ""``
- ``std::vector<double> _auc = {``

