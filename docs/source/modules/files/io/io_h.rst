io.h
====

**File Path**: ``modules/io/include/io/io.h``

**File Type**: H (Header)

**Lines**: 149

Dependencies
------------

**Includes**:

- ``H5Cpp.h``
- ``TBranch.h``
- ``TFile.h``
- ``TLeaf.h``
- ``TTree.h``
- ``TTreeReader.h``
- ``TTreeReaderArray.h``
- ``map``
- ``meta/meta.h``
- ``notification/notification.h``
- ``string``
- ``structs/element.h``
- ``structs/folds.h``
- ``structs/settings.h``
- ``tools/tools.h``

Classes
-------

``io``
~~~~~~

**Inherits from**: ``tools, 
    public notification``

**Methods**:

- ``void write(std::vector<g>* inpt, std::string set_name)``
- ``void write(g* inpt, std::string set_name)``
- ``void read(std::vector<g>* outpt, std::string set_name)``
- ``void read(g* out, std::string set_name)``
- ``void read(graph_hdf5_w* out, std::string set_name)``
- ``bool start(std::string filename, std::string read_write)``
- ``void end()``
- ``vector<std::string> dataset_names()``
- ``map<std::string, long> root_size()``
- ``void check_root_file_paths()``

