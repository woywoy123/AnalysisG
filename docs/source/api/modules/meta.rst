Metadata Module (C++)
=====================

The Metadata module provides metadata handling for ATLAS datasets.

Overview
--------

Located in ``src/AnalysisG/modules/meta/``, this module implements C++ metadata 
management:

- DSID (Dataset ID) lookups
- Cross-section information
- Generator filter efficiency
- Event counts and weights
- PyAMI integration

Purpose
-------

The meta module enables:

- Automatic metadata retrieval
- Dataset information caching
- Cross-section normalization
- Luminosity calculations

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/meta/cxx/*.cxx`` - Metadata implementations
- ``src/AnalysisG/modules/meta/include/meta/*.h`` - Metadata headers

**Python Binding**

- ``src/AnalysisG/core/meta.pyx`` - Cython wrapper
- ``src/AnalysisG/core/meta.pxd`` - Cython declarations

Key Classes
-----------

**Meta**

Metadata container:

.. code-block:: cpp

   class meta {
   public:
       // Dataset identification
       int dsid;
       std::string dataset_name;
       
       // Cross-section information
       double cross_section;        // pb
       double filter_efficiency;    // Generator filter eff
       double k_factor;             // QCD K-factor
       
       // Event information
       long total_events;           // Total events in dataset
       double sum_weights;          // Sum of event weights
       double sum_weights_squared;  // For uncertainty
       
       // Generator information
       std::string generator;       // Pythia, Sherpa, etc.
       std::string campaign;        // MC campaign (mc16a, mc16d, etc.)
       
       // File information
       std::vector<std::string> files;
       std::map<std::string, long> file_events;
   };

Features
--------

**Automatic Retrieval**

Fetch metadata from PyAMI:

.. code-block:: cpp

   meta m = fetch_metadata(dsid);

**Caching**

Cache metadata locally:

- JSON cache files
- Fast lookup
- Persistent storage
- Automatic updates

**Normalization**

Calculate normalization factors:

.. code-block:: cpp

   double lumi = 139.0;  // fb^-1
   double norm = (cross_section * filter_efficiency * k_factor * lumi) / total_events;

Usage Example
-------------

**C++ Usage**

.. code-block:: cpp

   #include <meta/meta.h>
   
   // Create metadata
   meta m;
   m.dsid = 410470;
   
   // Fetch from PyAMI
   if (m.fetch()) {
       std::cout << "Cross-section: " << m.cross_section << " pb" << std::endl;
       std::cout << "Filter eff: " << m.filter_efficiency << std::endl;
       std::cout << "Total events: " << m.total_events << std::endl;
   }
   
   // Calculate weight
   double lumi = 139.0;  // fb^-1
   double event_weight = m.get_weight(lumi);

**Python Usage**

.. code-block:: python

   from AnalysisG.core.meta import Meta, MetaData
   
   # Create metadata
   m = Meta()
   m.dsid = 410470
   
   # Fetch information
   if m.Fetch():
       print(f"Cross-section: {m.crossSection} pb")
       print(f"Filter eff: {m.genFiltEff}")
       print(f"Total events: {m.totalEvents}")

PyAMI Integration
-----------------

The module integrates with PyAMI (ATLAS Metadata Interface):

- Automatic DSID lookup
- Cross-section retrieval
- Generator information
- Campaign details

**Authentication**

PyAMI requires ATLAS credentials:

.. code-block:: bash

   voms-proxy-init -voms atlas
   ami auth

**Query Examples**

Common PyAMI queries:

.. code-block:: python

   # Get dataset info
   ami.get_dataset_info(dsid)
   
   # Get cross-section
   ami.get_cross_section(dsid)
   
   # Get provenance
   ami.get_provenance(dsid)

Metadata Fields
---------------

**Dataset Identification**

- DSID: Unique dataset identifier
- Dataset name: Full ATLAS dataset name
- Logical dataset name

**Physics Information**

- Cross-section (pb)
- Generator filter efficiency
- K-factor for QCD corrections
- PDF set

**Event Statistics**

- Total events
- Sum of weights
- Sum of weights squared
- Average weight

**Generator Details**

- Generator name (Pythia8, Sherpa, etc.)
- Generator version
- Tune parameters
- Parton shower

**Production Details**

- MC campaign (mc16a, mc16d, mc16e)
- AMI tag
- Production date
- ATLAS release

Cache Management
----------------

**Cache Location**

Default cache directory:

.. code-block:: cpp

   std::string cache_dir = "~/.analysisG/meta_cache/";

**Cache Format**

JSON format for easy inspection:

.. code-block:: json

   {
       "dsid": 410470,
       "cross_section": 831.76,
       "filter_efficiency": 1.0,
       "total_events": 50000000,
       "generator": "Pythia8EvtGen",
       "campaign": "mc16a"
   }

**Cache Operations**

.. code-block:: cpp

   // Save to cache
   m.save_cache();
   
   // Load from cache
   m.load_cache();
   
   // Clear cache
   meta::clear_cache();
   
   // Update cache
   m.update_cache();

Weight Calculations
-------------------

**Event Weight**

Calculate per-event weight:

.. code-block:: cpp

   double get_weight(double luminosity) {
       return (cross_section * filter_efficiency * k_factor * luminosity) / total_events;
   }

**Normalization**

Normalize histogram to luminosity:

.. code-block:: cpp

   double normalize(double lumi, long n_events_processed) {
       return (cross_section * filter_efficiency * lumi) / n_events_processed;
   }

Error Handling
--------------

Handle metadata retrieval errors:

- DSID not found
- PyAMI connection failure
- Invalid cross-section
- Missing information
- Cache corruption

See Also
--------

* :doc:`../core/meta` - Python Meta wrapper
* :doc:`analysis` - Analysis using metadata
* :doc:`io` - I/O operations
