Analysis Module (C++)
=====================

The Analysis module provides the core execution engine for AnalysisG analyses.

Overview
--------

Located in ``src/AnalysisG/modules/analysis/``, this module implements the C++ backend 
that orchestrates the analysis workflow, including:

- Sample management and file processing
- Event loop execution
- Template instantiation and management
- Multi-threading coordination
- Progress tracking and reporting

C++ Implementation
------------------

The analysis engine is implemented in C++ for maximum performance and integrates with:

- Event templates for data loading
- Graph templates for GNN construction
- Selection templates for event filtering
- Model templates for ML inference
- Metric templates for evaluation

Key Components
--------------

**Analysis Engine**

The ``analysis`` class in ``src/AnalysisG/modules/analysis/`` provides:

- ``start()``: Begin analysis execution
- ``add_samples()``: Register input ROOT files
- ``add_event_template()``: Register event definitions
- ``add_graph_template()``: Register graph definitions
- ``add_selection_template()``: Register selection criteria
- ``add_model()``: Register ML models for training/inference
- ``add_metric_template()``: Register evaluation metrics

**Threading Model**

The analysis engine supports multi-threaded execution:

- Parallel file processing
- Thread-safe template instantiation
- Concurrent event processing
- Progress synchronization across threads

**Progress Reporting**

Real-time progress tracking:

- ``progress()``: Get current progress state
- ``progress_mode()``: Get current processing mode
- ``progress_report()``: Get detailed progress information
- ``is_complete()``: Check completion status

Integration with Python
-----------------------

The C++ analysis engine is exposed to Python via Cython wrappers in 
``src/AnalysisG/core/analysis.pyx``. See :doc:`../core/analysis` for Python API.

Usage Pattern
-------------

The typical workflow:

1. Create analysis instance
2. Add input samples with ``add_samples()``
3. Register event template with ``add_event_template()``
4. Register optional graph/selection/model templates
5. Call ``start()`` to begin execution
6. Monitor progress via progress methods

Performance Considerations
--------------------------

The C++ implementation provides:

- Zero-copy data access where possible
- Efficient memory management
- Parallel processing of independent files
- Optimized event loop execution

See Also
--------

* :doc:`../core/analysis` - Python API wrapper
* :doc:`event` - Event template implementation
* :doc:`graph` - Graph template implementation
* :doc:`selection` - Selection template implementation
