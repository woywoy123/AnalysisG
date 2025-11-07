container Module
================

Location
--------

``src/AnalysisG/modules/container``

Overview
--------

This module contains 3 files implementing the container functionality.

Files
-----


container.cxx
^^^^^^^^^^^^^

**Path**: ``modules/container/cxx/container.cxx``


entries.cxx
^^^^^^^^^^^

**Path**: ``modules/container/cxx/entries.cxx``


container.h
^^^^^^^^^^^

**Path**: ``modules/container/include/container/container.h``

**Classes**:

- ``container`` (inherits from ``tools``)

**Structs**: ``entry_t``

**Functions** (sample):

- ``void init()``
- ``void destroy()``
- ``bool has_event(event_template* ev)``
- ``bool has_graph(graph_template* gr)``
- ``bool has_selection(selection_template* sel)``

