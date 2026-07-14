Analysis Module
===============

``analysis`` is the top-level pipeline class.  It accepts user-defined event,
graph, selection, model, and metric templates; reads ROOT and HDF5 files; builds
all compiled objects in parallel; and launches the training/evaluation loop.
It inherits from ``notification`` (logging) and ``tools`` (utilities).

Class: ``analysis``
--------------------

**Header:** ``<AnalysisG/analysis.h>``

**Inheritance:** ``notification``, ``tools``

Configuration Field
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Description
   * - ``m_settings``
     - ``settings_t``
     - Aggregated settings struct controlling output paths, thread counts,
       batch size, kfolds, epochs, TrainSize, etc.
   * - ``meta_data``
     - ``std::map<std::string, meta*>``
     - Dataset metadata objects indexed by file path.

Registration Methods
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``void add_samples(std::string path, std::string label)``
     - Registers a ROOT file or directory *path* under dataset label *label*.
   * - ``void add_event_template(event_template* ev, std::string label)``
     - Registers an event-class prototype *ev* for files with label *label*.
   * - ``void add_graph_template(graph_template* gr, std::string label)``
     - Registers a graph-class prototype *gr* for label *label*.
   * - ``void add_selection_template(selection_template* sel)``
     - Registers a selection prototype (applied to all events).
   * - ``void add_metric_template(metric_template* mx, model_template* mdl)``
     - Associates a metric *mx* with model *mdl*.
   * - ``void add_model(model_template* model, optimizer_params_t* op, std::string run_name)``
     - Registers a model with optimiser parameters and run name.
   * - ``void add_model(model_template* model, std::string run_name)``
     - Registers a model with default optimiser parameters.

Execution Methods
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``void attach_threads()``
     - Attaches the thread pool to the event/graph compilation workers.
   * - ``void start()``
     - Runs the complete pipeline: reads files → builds events → builds
       graphs → builds selections → trains models → evaluates metrics.
   * - ``std::map<std::string, std::vector<float>> progress()``
     - Returns per-model training progress as a map of model-name →
       list of (epoch, loss) pairs.
   * - ``std::map<std::string, std::string> progress_mode()``
     - Returns the current training mode string per model.
   * - ``std::map<std::string, std::string> progress_report()``
     - Returns a human-readable progress report per model.
   * - ``std::map<std::string, bool> is_complete()``
     - Returns whether each model has finished all epochs.
