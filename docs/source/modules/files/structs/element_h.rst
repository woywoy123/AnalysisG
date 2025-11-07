element.h
=========

**File Path**: ``modules/structs/include/structs/element.h``

**File Type**: H (Header)

**Lines**: 124

Dependencies
------------

**Includes**:

- ``TBranch.h``
- ``TFile.h``
- ``TLeaf.h``
- ``TTree.h``
- ``TTreeReader.h``
- ``TTreeReaderArray.h``
- ``iostream``
- ``map``
- ``string``
- ``structs/base.h``
- ``structs/meta.h``
- ``tools/vector_cast.h``
- ``vector``

Structs
-------

``data_t``
~~~~~~~~~~

``element_t``
~~~~~~~~~~~~~

**Members**:

- ``std::string tree = ""``
- ``bool next()``
- ``void set_meta()``
- ``long event_index = -1``
- ``std::string filename = ""``
- ``bool boundary()``
- ``template <typename g>
    bool get(std::string key, g* var){
        if (!this -> handle.count(key)){return false``

``write_t``
~~~~~~~~~~~

**Members**:

- ``TFile* file = nullptr``
- ``TTree* tree = nullptr``
- ``meta_t* mtx = nullptr``
- ``std::map<std::string, variable_t*>* data = nullptr``
- ``variable_t* process(std::string* name)``
- ``void write()``
- ``void create(std::string tr_name, std::string path)``
- ``void close()``

``writer``
~~~~~~~~~~

**Members**:

- ``public:
        writer()``
- ``~writer()``
- ``void create(std::string* pth)``
- ``void write(std::string* tree)``
- ``template <typename g>
        void process(std::string* tree, std::string* name, g* t){
            this -> process(tree, name) -> process(t, name, nullptr)``

