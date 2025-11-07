meta.h
======

**File Path**: ``modules/meta/include/meta/meta.h``

**File Type**: H (Header)

**Lines**: 178

Dependencies
------------

**Includes**:

- ``TBranch.h``
- ``TFile.h``
- ``TH1F.h``
- ``TLeaf.h``
- ``TTree.h``
- ``notification/notification.h``
- ``rapidjson/document.h``
- ``structs/folds.h``
- ``structs/meta.h``
- ``structs/property.h``
- ``tools/tools.h``

Classes
-------

``meta``
~~~~~~~~

**Inherits from**: ``tools,
    public notification``

**Methods**:

- ``void scan_data(TObject* obj)``
- ``void scan_sow(TObject* obj)``
- ``void parse_json(std::string inpt)``
- ``string hash(std::string fname)``
- ``void compiler()``
- ``float parse_float(std::string key, TTree* tr)``
- ``string parse_string(std::string key, TTree* tr)``
- ``static get_isMC(bool*, meta*)``
- ``static get_found(bool*, meta*)``
- ``static get_eventNumber(double*, meta*)``

