Meta Module
==========

The Meta module is responsible for managing metadata and configuration settings for analysis tasks in the AnalysisG framework.

Overview
--------

This module stores and processes metadata from ROOT files, including information fetched via PyAMI. It provides a centralized approach for accessing physics dataset information, run configurations, and experiment parameters.

Key Components
-------------

meta class
~~~~~~~~

.. doxygenclass:: meta
   :members:
   :protected-members:
   :undoc-members:

meta_t struct
~~~~~~~~~~

.. doxygenstruct:: meta_t
   :members:
   :undoc-members:

Main Functionalities
-------------------

Metadata Management
~~~~~~~~~~~~~~~~

The module provides methods for storing and retrieving dataset metadata:

- Storage of dataset information such as cross-sections, k-factors, and event counts
- Access to production and processing metadata for physics datasets
- Configuration parameters for analysis workflows

JSON Configuration
~~~~~~~~~~~~~~~

Support for parsing and handling configuration from JSON files:

- ``parse_json()``: Parses JSON configuration data
- Storage of structured configuration parameters
- Access to configuration through property interfaces

ROOT Metadata Extraction
~~~~~~~~~~~~~~~~~~~~~

Functionality for extracting metadata from ROOT files:

- ``scan_data()``: Scans and extracts metadata from ROOT files
- ``parse_float()``, ``parse_string()``: Parse type-specific values from metadata
- Mapping of metadata to internal structures

Property System
~~~~~~~~~~~~

The module implements a property system for metadata access:

- Properties for numerical values like cross sections, luminosity, energy
- Properties for string values like dataset names, generator information
- Automatic property mapping to internal metadata structure

Usage Example
------------

.. code-block:: cpp

    #include <meta/meta.h>
    
    void metadata_example(TFile* root_file) {
        meta* metadata = new meta();
        
        // Scan metadata from a ROOT file
        TObject* file_metadata = root_file->Get("Metadata");
        metadata->scan_data(file_metadata);
        
        // Access metadata properties
        double cross_section = metadata->cross_section_pb;
        double lumi = metadata->campaign_luminosity;
        std::string dataset = metadata->DatasetName;
        
        // Create a unique hash for the dataset
        std::string hash_id = metadata->hash(dataset);
        
        // Get fold tags for the dataset
        const folds_t* tags = metadata->get_tags(hash_id);
        
        delete metadata;
    }