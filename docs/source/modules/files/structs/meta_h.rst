meta.h
======

**File Path**: ``modules/structs/include/structs/meta.h``

**File Type**: H (Header)

**Lines**: 97

Dependencies
------------

**Includes**:

- ``iostream``
- ``map``
- ``string``
- ``vector``

Structs
-------

``weights_t``
~~~~~~~~~~~~~

**Members**:

- ``int dsid = -1``
- ``bool isAFII = false``
- ``std::string generator = ""``
- ``std::string ami_tag = ""``
- ``float total_events_weighted = -1``
- ``float total_events = -1``
- ``float processed_events = -1``
- ``float processed_events_weighted = -1``
- ``float processed_events_weighted_squared = -1``
- ``std::map<std::string, float> hist_data = {``

``meta_t``
~~~~~~~~~~

**Members**:

- ``bool isMC = true``
- ``std::string derivationFormat = ""``
- ``std::map<int, std::string> inputfiles = {``

