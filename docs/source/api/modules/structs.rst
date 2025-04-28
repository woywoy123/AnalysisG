Structs Module
=============

The Structs module provides fundamental data structures used throughout the AnalysisG framework.

Overview
--------

This module defines a collection of data structures, type definitions, and enumerations that form the foundation of the AnalysisG framework. It provides standardized containers and types that ensure consistency across the framework.

Key Components
-------------

settings_t
~~~~~~~~

.. doxygenstruct:: settings_t
   :members:
   :undoc-members:

graph_t
~~~~~

.. doxygenstruct:: graph_t
   :members:
   :undoc-members:

event_t
~~~~~

.. doxygenstruct:: event_t
   :members:
   :undoc-members:

Main Functionalities
-------------------

Core Data Structures
~~~~~~~~~~~~~~~~~

The module defines core data structures used across the framework:

- Event representations and containers
- Graph structures for GNN implementations
- Settings and configuration containers
- Feature storage and management structures

Type Definitions
~~~~~~~~~~~~

Standard type definitions to ensure consistency:

- Specialized container types for physics objects
- Type aliases for common data types
- Consistent naming conventions across the framework

Enumeration Types
~~~~~~~~~~~~~

Common enumeration types used throughout the framework:

- Mode enumerations (training, validation, testing)
- Status and error codes
- Feature type classifications
- Physics-specific enumerations

Configuration Structures
~~~~~~~~~~~~~~~~~~~~

Structures for configuring various components:

- Analysis configuration parameters
- Model and training settings
- IO and data handling settings
- Feature extraction parameters

Usage Example
------------

.. code-block:: cpp

    #include <structs/structs.h>
    
    void configure_analysis() {
        // Create and configure analysis settings
        settings_t settings;
        
        // General settings
        settings.run_name = "top_classification";
        settings.output_path = "./results";
        
        // Model settings
        settings.model_name = "GNN";
        settings.learning_rate = 0.001;
        settings.batch_size = 128;
        settings.epochs = 50;
        
        // Data settings
        settings.train_fraction = 0.7;
        settings.validation_fraction = 0.15;
        settings.test_fraction = 0.15;
        settings.input_files = {"data1.root", "data2.root"};
        
        // Feature settings
        settings.node_features = {"pt", "eta", "phi", "energy"};
        settings.edge_features = {"deltaR", "deltaEta", "deltaPhi"};
        settings.graph_features = {"total_pt", "multiplicity"};
        
        // Pass settings to analysis object
        analysis* an = new analysis();
        an->import_settings(&settings);
    }