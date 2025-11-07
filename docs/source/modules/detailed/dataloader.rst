dataloader Module
=================

Location
--------

``src/AnalysisG/modules/dataloader``

Overview
--------

This module contains 4 files implementing the dataloader functionality.

Files
-----


cache.cxx
^^^^^^^^^

**Path**: ``modules/dataloader/cxx/cache.cxx``


dataloader.cxx
^^^^^^^^^^^^^^

**Path**: ``modules/dataloader/cxx/dataloader.cxx``


dataset.cxx
^^^^^^^^^^^

**Path**: ``modules/dataloader/cxx/dataset.cxx``


dataloader.h
^^^^^^^^^^^^

**Path**: ``modules/dataloader/include/generators/dataloader.h``

**Classes**:

- ``analysis``
- ``model_template``
- ``dataloader`` (inherits from ``notification, 
    public tools``)
- ``analysis``

**Functions** (sample):

- ``void safe_delete(std::vector<graph_t*>* data)``
- ``void generate_test_set(float percentage = 50)``
- ``void generate_kfold_set(int k)``
- ``void dump_dataset(std::string path)``
- ``bool restore_dataset(std::string path)``

