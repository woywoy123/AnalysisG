Structs and Enumerations
========================

Framework-wide plain-old-data structures and enumeration types.  These types
are used pervasively across the C++ core and are the primary data contracts
between modules (e.g. between the IO layer and the GNN training pipeline).

Enumerations
------------

.. doxygenenum:: data_enum
   :project: AnalysisG

.. doxygenenum:: opt_enum
   :project: AnalysisG

.. doxygenenum:: mlp_init
   :project: AnalysisG

.. doxygenenum:: loss_enum
   :project: AnalysisG

.. doxygenenum:: scheduler_enum
   :project: AnalysisG

.. doxygenenum:: graph_enum
   :project: AnalysisG

.. doxygenenum:: mode_enum
   :project: AnalysisG

.. doxygenenum:: particle_enum
   :project: AnalysisG

Core Data Structs
-----------------

.. rubric:: bsc_t — polymorphic leaf buffer

``bsc_t`` (base struct) is the polymorphic root of the ROOT leaf-reading
hierarchy.  Each instantiation holds at most one heap-allocated buffer
corresponding to the concrete ``data_enum`` type discovered at runtime.
All overloads of ``element()`` read the buffer at ``index`` and write into
the caller-supplied pointer.

.. doxygenstruct:: bsc_t
   :project: AnalysisG
   :members:

.. rubric:: data_t — single ROOT leaf accessor

``data_t`` extends ``bsc_t`` with ROOT bookkeeping (TLeaf/TBranch/TTree
pointers, file path, leaf type string) and sequential iteration helpers.

.. doxygenstruct:: data_t
   :project: AnalysisG
   :members:

.. rubric:: element_t — per-event leaf handle map

``element_t`` is handed to ``particle_template::build()`` and
``event_template::build()``.  The ``get<T>(key, ptr)`` template method
looks up the named ``data_t`` and copies the current element into ``*ptr``
via the appropriate ``bsc_t::element()`` overload.

.. doxygenstruct:: element_t
   :project: AnalysisG
   :members:

.. rubric:: write_t / writer — ROOT output helpers

.. doxygenstruct:: write_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: writer
   :project: AnalysisG
   :members:

Event, Particle, and Graph Payload Structs
------------------------------------------

.. rubric:: particle_t — raw kinematic payload

``particle_t`` carries the floating-point kinematics and integer metadata
for a single particle.  ``particle_template`` stores one of these as its
internal representation.

.. doxygenstruct:: particle_t
   :project: AnalysisG
   :members:

.. rubric:: event_t — event identity

``event_t`` is embedded in every ``event_template`` and carries the event
index, weight, ROOT tree name and the unique hash used for graph caching.

.. doxygenstruct:: event_t
   :project: AnalysisG
   :members:

.. rubric:: graph_t — GNN tensor container

``graph_t`` is the central data structure passed between the graph builder,
the dataloader, and the ``model_template::forward()`` method.  It stores
batched PyTorch tensors for node/edge/graph data features, truth features,
the COO edge index, and batching meta-data.

.. doxygenstruct:: graph_t
   :project: AnalysisG
   :members:

.. rubric:: folds_t — k-fold assignment

.. doxygenstruct:: folds_t
   :project: AnalysisG
   :members:

.. rubric:: graph_hdf5 / graph_hdf5_w — HDF5 serialisation records

.. doxygenstruct:: graph_hdf5
   :project: AnalysisG
   :members:

.. doxygenstruct:: graph_hdf5_w
   :project: AnalysisG
   :members:

Settings and Configuration Structs
-----------------------------------

.. rubric:: settings_t — global analysis settings

``settings_t`` is the POD configuration object stored inside ``analysis``
(and accessible from Python as properties).  All ``Analysis.*`` properties
map onto fields of this struct.

.. doxygenstruct:: settings_t
   :project: AnalysisG
   :members:

.. rubric:: model_settings_t — per-model ML configuration

``model_settings_t`` is populated by ``model_template`` and carries
optimizer choice, I/O feature maps, weight/tree names, and device info.

.. doxygenstruct:: model_settings_t
   :project: AnalysisG
   :members:

.. rubric:: loss_opt — loss function options

.. doxygenstruct:: loss_opt
   :project: AnalysisG
   :members:

.. rubric:: optimizer_params_t — optimizer hyper-parameters

``optimizer_params_t`` is the C++ counterpart of ``OptimizerConfig``
(Cython layer).  Each ``cproperty`` field sets a ``m_*`` sentinel flag so
the optimizer builder knows which hyper-parameters have been explicitly
specified.

.. doxygenclass:: optimizer_params_t
   :project: AnalysisG
   :members:

Meta Structs
------------

.. rubric:: meta_t — ATLAS dataset metadata

``meta_t`` holds all AMI / ATLAS metadata for a dataset: DSID, campaign,
generator, cross-section, filter efficiency, luminosity, sum-of-weights,
run numbers, file GUIDs and per-systematic weight dictionaries.

.. doxygenstruct:: meta_t
   :project: AnalysisG
   :members:

.. rubric:: weights_t — per-systematic sum-of-weights record

.. doxygenstruct:: weights_t
   :project: AnalysisG
   :members:

Training Report Structs
-----------------------

.. rubric:: model_report — per-epoch training summary

``model_report`` is produced by the dataloader after each epoch and
carries loss/accuracy maps keyed by ``mode_enum`` and feature name, as
well as the current learning rates and iteration counters.

.. doxygenstruct:: model_report
   :project: AnalysisG
   :members:

.. rubric:: roc_t — ROC curve data

.. doxygenstruct:: roc_t
   :project: AnalysisG
   :members:
