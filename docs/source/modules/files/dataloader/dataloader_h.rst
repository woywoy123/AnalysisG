dataloader.h
============

**File Path**: ``modules/dataloader/include/generators/dataloader.h``

**File Type**: H (Header)

**Lines**: 103

Dependencies
------------

**Includes**:

- ``algorithm``
- ``c10/cuda/CUDACachingAllocator.h``
- ``cuda.h``
- ``map``
- ``notification/notification.h``
- ``random``
- ``structs/property.h``
- ``structs/settings.h``
- ``templates/graph_template.h``
- ``tools/tools.h``

Classes
-------

``dataloader``
~~~~~~~~~~~~~~

**Inherits from**: ``notification, 
    public tools``

**Methods**:

- ``void safe_delete(std::vector<graph_t*>* data)``
- ``void generate_test_set(float percentage = 50)``
- ``void generate_kfold_set(int k)``
- ``void dump_dataset(std::string path)``
- ``bool restore_dataset(std::string path)``
- ``vector<graph_t*> get_random(int num = 5)``
- ``void extract_data(graph_t* gr)``
- ``void datatransfer(torch::TensorOptions* op, size_t* num_events = nul...)``
- ``void datatransfer(std::map<int, torch::TensorOptions*>* ops)``
- ``bool dump_graphs(std::string path = "./", int threads = 10)``

Structs
-------

``model_report``
~~~~~~~~~~~~~~~~

