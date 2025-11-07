analysis Module
===============

Location
--------

``src/AnalysisG/modules/analysis``

Overview
--------

This module contains 9 files implementing the analysis functionality.

Files
-----


analysis.cxx
^^^^^^^^^^^^

**Path**: ``modules/analysis/analysis.cxx``


event_build.cxx
^^^^^^^^^^^^^^^

**Path**: ``modules/analysis/event_build.cxx``


graph_build.cxx
^^^^^^^^^^^^^^^

**Path**: ``modules/analysis/graph_build.cxx``


analysis.h
^^^^^^^^^^

**Path**: ``modules/analysis/include/AnalysisG/analysis.h``

**Classes**:

- ``analysis`` (inherits from ``notification, 
    public tools``)

**Functions** (sample):

- ``void flush(std::map<std::string, g*>* data)``
- ``void add_samples(std::string path, std::string label)``
- ``void add_selection_template(selection_template* sel)``
- ``void add_event_template(event_template* ev, std::string label)``
- ``void add_graph_template(graph_template* gr, std::string label)``


inference_build.cxx
^^^^^^^^^^^^^^^^^^^

**Path**: ``modules/analysis/inference_build.cxx``


methods.cxx
^^^^^^^^^^^

**Path**: ``modules/analysis/methods.cxx``


metric_build.cxx
^^^^^^^^^^^^^^^^

**Path**: ``modules/analysis/metric_build.cxx``


optimizer_build.cxx
^^^^^^^^^^^^^^^^^^^

**Path**: ``modules/analysis/optimizer_build.cxx``


selection_build.cxx
^^^^^^^^^^^^^^^^^^^

**Path**: ``modules/analysis/selection_build.cxx``

