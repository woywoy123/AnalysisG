sampletracer.h
==============

**File Path**: ``modules/sampletracer/include/generators/sampletracer.h``

**File Type**: H (Header)

**Lines**: 35

Dependencies
------------

**Includes**:

- ``container/container.h``
- ``notification/notification.h``
- ``thread``

Classes
-------

``sampletracer``
~~~~~~~~~~~~~~~~

**Inherits from**: ``tools, 
    public notification``

**Methods**:

- ``bool add_meta_data(meta* meta_, std::string filename)``
- ``vector<event_template*> get_events(std::string label)``
- ``void fill_selections(std::map<std::string, selection_template*>* inpt)``
- ``bool add_event(event_template* ev, std::string label)``
- ``bool add_graph(graph_template* gr, std::string label)``
- ``bool add_selection(selection_template* sel)``
- ``void populate_dataloader(dataloader* dl)``
- ``void compile_objects(int threads)``

