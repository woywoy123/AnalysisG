```rst
AnalysisG Modules Documentation
===============================

Overview
--------

The ``modules`` directory contains the core functionality of the AnalysisG framework.
It is organized into several subdirectories, each implementing specific functionality
for particle physics analysis, data processing, and machine learning applications.

Directory Structure
-------------------

Based on the ``CMakeLists.txt``, the modules directory contains the following subdirectories:

-   ``typecasting``: Type conversion utilities
-   ``notification``: User notification system
-   ``structs``: Data structures and enumerations
-   ``tools``: General purpose utility functions
-   ``particle``: Particle physics object representations
-   ``event``: Event data handling
-   ``selection``: Event selection criteria
-   ``graph``: Graph-based data representations for machine learning
-   ``model``: Machine learning model implementations
-   ``metrics``: Performance metrics collection
-   ``metric``: Individual metric calculations
-   ``meta``: Metadata handling
-   ``io``: Input/Output operations
-   ``plotting``: Data visualization
-   ``lossfx``: Loss functions for model training
-   ``container``: Data container implementations
-   ``sampletracer``: Sample tracking and progress monitoring
-   ``analysis``: Analysis workflow coordination
-   ``dataloader``: Data loading utilities
-   ``optimizer``: Model optimization implementations

Key Components
--------------

Structs Module
~~~~~~~~~~~~~~

The ``structs`` module provides fundamental data structures and enumerations used throughout the framework. It includes:

-   ``enums.h``: Defines enumerations like ``data_enum`` for type identification
-   ``base.h/base.cxx``: Provides the ``bsc_t`` class with core functionality for data type handling
-   ``model.h``: Defines the ``model_settings_t`` structure for configuring ML models

The ``root_type_translate`` function in ``base.cxx`` is essential for converting between ROOT data types and internal representations.

Model Module
~~~~~~~~~~~~

The ``model`` module implements machine learning model templates and functionality:

-   ``model_template.cxx``: Defines the base class for all models with methods for:

    -   Feature assignment
    -   Module registration
    -   Forward propagation
    -   Memory management

Analysis Module
~~~~~~~~~~~~~~~

The ``analysis`` module coordinates the overall analysis workflow:

-   ``optimizer_build.cxx``: Contains methods for setting up model training sessions
-   Manages k-fold cross-validation
-   Handles data transfer between devices
-   Provides progress tracking mechanisms

Event Module
~~~~~~~~~~~~

The ``event_template`` class in the event module manages particle physics event data:

-   Tree and branch handling
-   Event weighting
-   Event indexing
-   Hash-based identification

SampleTracer Module
~~~~~~~~~~~~~~~~~~~

The ``sampletracer`` module provides progress tracking and reporting:

-   ``compile_objects``: Processes data objects with multi-threading support
-   Progress visualization during long-running operations

IO Module
~~~~~~~

The ``io`` module handles input/output operations:

-   File management
-   Configuration loading
-   Integration with ROOT file system

Code Examples
-------------

Data Type Translation
~~~~~~~~~~~~~~~~~~~~~

The ``root_type_translate`` function in ``base.cxx`` converts between ROOT data types and internal enums::

    data_enum bsc_t::root_type_translate(std::string* root_str){
        int vec = count(root_str, "vector");
        if (vec == 0 && (*root_str) ==   "Float_t"){return data_enum::v_f  ;}
        if (vec == 0 && (*root_str) ==  "Double_t"){return data_enum::v_d  ;}
        // Additional type mappings...
    }

Model Training Session Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    void analysis::build_model_session(){
        // Set up k-fold cross-validation
        std::vector<int> kfold = this -> m_settings.kfold;
        if (!kfold.size()){
            for (int k(0); k < this -> m_settings.kfolds; ++k){
                kfold.push_back(k);
            }
        }

        // Set up device mapping
        std::map<int, torch::TensorOptions*> dev_map;
        // Transfer data to appropriate devices
        this -> loader -> datatransfer(&dev_map);

        // Create optimizer instances for each model session
        for (size_t x(0); x < this -> model_sessions.size(); ++x){
            std::string name = this -> model_session_names[x];
            optimizer* optim = new optimizer();
            // Configure and initialize the optimizer
            // ...
        }
    }

Integration Points
------------------

The modules are designed to work together through well-defined interfaces:

1.  ``structs`` provide the fundamental data types used by all other modules
2.  ``model`` templates use ``graph`` objects for data representation
3.  ``analysis`` coordinates ``optimizer`` instances to train models
4.  ``sampletracer`` monitors the progress of operations
5.  ``io`` handles data persistence and loading

Adding New Functionality
------------------------

To extend the framework:

1.  For new data types: Add entries to the appropriate enums and update type translation functions
2.  For new models: Subclass ``model_template`` and implement the required methods
3.  For new analysis techniques: Add new methods to the ``analysis`` class
4.  For new metrics: Add implementations to the ``metrics`` module

Building and Compilation
------------------------

The modules are built using CMake, as specified in ``CMakeLists.txt``.
Configuration files are generated for the ``analysis`` and ``io`` modules.
```