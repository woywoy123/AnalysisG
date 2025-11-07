container.h
===========

**File Path**: ``modules/container/include/container/container.h``

**File Type**: H (Header)

**Lines**: 66

Dependencies
------------

**Includes**:

- ``generators/dataloader.h``
- ``meta/meta.h``
- ``templates/event_template.h``
- ``templates/graph_template.h``
- ``templates/selection_template.h``
- ``tools/tools.h``

Classes
-------

``container``
~~~~~~~~~~~~~

**Inherits from**: ``tools``

**Methods**:

- ``void add_meta_data(meta*, std::string)``
- ``bool add_selection_template(selection_template*)``
- ``bool add_event_template(event_template*, std::string label)``
- ``bool add_graph_template(graph_template*, std::string label)``
- ``void fill_selections(std::map<std::string, selection_template*>* inpt)``
- ``void get_events(std::vector<event_template*>*, std::string label)``
- ``void populate_dataloader(dataloader* dl)``
- ``void compile(size_t* len, int threadIdx)``
- ``size_t len()``

Structs
-------

``entry_t``
~~~~~~~~~~~

**Members**:

- ``std::string hash = ""``
- ``std::vector<graph_t*>                m_data  = {``

