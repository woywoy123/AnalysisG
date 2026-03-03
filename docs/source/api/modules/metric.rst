Metric Template
===============

``metric_template`` is the base class for all user-defined training metrics.
It inherits from both ``tools`` and ``notification``, and provides helpers for
computing per-event and per-batch statistics during model training, validation,
and evaluation.  It also exposes ROOT tree output helpers (``register_output``,
``write``) and the same particle combination utilities as ``selection_template``.

Struct: ``metric_t``
---------------------

**Header:** ``<templates/metric_template.h>``

``metric_t`` is constructed by the framework and passed to
``metric_template::define_metric`` for each graph in the batch.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Description
   * - ``kfold``
     - ``int``
     - Current k-fold index.
   * - ``epoch``
     - ``int``
     - Current training epoch.
   * - ``device``
     - ``int``
     - CUDA device index (0 = first GPU, -1 = CPU).

Public Method
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``template<g> g get(graph_enum grx, std::string name)``
     - Retrieves the named graph/node/edge variable of type *g* from the
       variable store for this event. Prints an error and returns default
       if the variable is not found.
   * - ``std::string mode()``
     - Returns the training mode string (``"train"``, ``"valid"``, ``"eval"``).
   * - ``std::string* get_filename(long unsigned int idx)``
     - Returns a pointer to the filename of graph at batch position *idx*.

Class: ``metric_template``
---------------------------

**Header:** ``<templates/metric_template.h>``

**Inheritance:** ``tools``, ``notification``

Properties
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Property
     - Type
     - Description
   * - ``name``
     - ``std::string``
     - Metric class name.  Settable/Gettable.
   * - ``output_path``
     - ``std::string``
     - Directory for ROOT output files.  Gettable.
   * - ``variables``
     - ``std::vector<std::string>``
     - List of variable names requested from ``graph_t`` (formatted as
       ``"graph_enum/feature_name"``, e.g. ``"data_node/pt"``).
   * - ``run_names``
     - ``std::map<std::string, std::string>``
     - Maps model-name → run-label used for output file organisation.

Virtual Methods (Override in Subclass)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``virtual metric_template* clone()``
     - Returns a heap-allocated copy of the metric.
   * - ``virtual void define_variables()``
     - **Called once before training.** Use ``variables`` to request which
       graph/node/edge features to expose in ``metric_t::get``.
   * - ``virtual void define_metric(metric_t* v)``
     - **Called once per graph** during training/validation/evaluation.
       Use ``v->get<T>(graph_enum, "name")`` to read features and compute
       derived quantities.
   * - ``virtual void event()``
     - Called after ``define_metric`` for each event.
   * - ``virtual void batch()``
     - Called after a full batch is processed.
   * - ``virtual void end()``
     - Called at the end of an epoch.

Output Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``template<T> void register_output(std::string tree, std::string name, T* t)``
     - Creates an output ROOT branch ``tree/name`` backed by *t*.
       Must be called in ``define_variables`` (once per run).
   * - ``template<T> void write(std::string tree, std::string name, T* t, bool fill = false)``
     - Fills the branch ``tree/name`` with current value of *t*.
       Pass ``fill=true`` to also flush the tree entry.

Particle Utilities (same as ``selection_template``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``metric_template`` exposes the same ``sum``, ``safe_delete``, ``make_unique``,
``vectorize`` helpers as ``selection_template``; see that page for documentation.

Additionally:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``std::vector<particle_template*> make_particle(...)``
     - Constructs a vector of ``particle_template`` objects from four
       ``std::vector<std::vector<double>>`` arrays of (pt, eta, phi, E)
       in cylindrical coordinates (MeV).
