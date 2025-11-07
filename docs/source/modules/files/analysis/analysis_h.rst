analysis.h
==========

**File Path**: ``modules/analysis/include/AnalysisG/analysis.h``

**File Type**: H (Header)

**Lines**: 124

Dependencies
------------

**Includes**:

- ``generators/dataloader.h``
- ``generators/optimizer.h``
- ``generators/sampletracer.h``
- ``io/io.h``
- ``string``
- ``structs/settings.h``
- ``templates/event_template.h``
- ``templates/graph_template.h``
- ``templates/metric_template.h``
- ``templates/model_template.h``
- ``templates/selection_template.h``

Classes
-------

``analysis``
~~~~~~~~~~~~

**Inherits from**: ``notification, 
    public tools``

**Methods**:

- ``void add_samples(std::string path, std::string label)``
- ``void add_selection_template(selection_template* sel)``
- ``void add_event_template(event_template* ev, std::string label)``
- ``void add_graph_template(graph_template* gr, std::string label)``
- ``void add_metric_template(metric_template* mx, model_template* mdl)``
- ``void add_model(model_template* model, optimizer_params_t* op, std...)``
- ``void add_model(model_template* model, std::string run_name)``
- ``void attach_threads()``
- ``void start()``
- ``map<std::string, std::string> progress_mode()``

Functions
---------

``void flush(std::map<std::string, g*>* data)``

