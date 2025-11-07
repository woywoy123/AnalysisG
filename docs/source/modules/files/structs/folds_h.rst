folds.h
=======

**File Path**: ``modules/structs/include/structs/folds.h``

**File Type**: H (Header)

**Lines**: 75

Dependencies
------------

**Includes**:

- ``string``

Structs
-------

``folds_t``
~~~~~~~~~~~

**Members**:

- ``int k = -1``
- ``bool is_train = false``
- ``bool is_valid = false``
- ``bool is_eval = false``
- ``char* hash = nullptr``
- ``void flush_data(){
        if (!this -> hash){return``

``graph_hdf5``
~~~~~~~~~~~~~~

**Members**:

- ``int    num_nodes = -1``
- ``double event_weight = 1``
- ``long   event_index = -1``
- ``std::string hash``
- ``std::string filename``
- ``std::string edge_index``
- ``std::string data_map_graph``
- ``std::string data_map_node``
- ``std::string data_map_edge``
- ``std::string truth_map_graph``

``graph_hdf5_w``
~~~~~~~~~~~~~~~~

**Members**:

- ``int    num_nodes = -1``
- ``double event_weight = 1``
- ``long   event_index = -1``
- ``char* hash = nullptr``
- ``char* filename = nullptr``
- ``char* edge_index = nullptr``
- ``char* data_map_graph = nullptr``
- ``char* data_map_node = nullptr``
- ``char* data_map_edge = nullptr``
- ``char* truth_map_graph = nullptr``

