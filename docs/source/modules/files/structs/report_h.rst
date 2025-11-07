report.h
========

**File Path**: ``modules/structs/include/structs/report.h``

**File Type**: H (Header)

**Lines**: 37

Dependencies
------------

**Includes**:

- ``map``
- ``string``
- ``structs/enums.h``

Structs
-------

``model_report``
~~~~~~~~~~~~~~~~

**Members**:

- ``int k``
- ``int epoch``
- ``bool is_complete = false``
- ``metrics* waiting_plot = nullptr``
- ``std::vector<double> current_lr = {``

