DataLoader Module (C++)
=======================

The DataLoader module provides data loading utilities for analysis workflows.

Overview
--------

Located in ``src/AnalysisG/modules/dataloader/``, this module implements C++ data 
loading functionality:

- Batch loading from ROOT files
- Data preprocessing pipelines
- Multi-threaded data loading
- Integration with training workflows

Purpose
-------

The dataloader module enables efficient data loading for:

- Machine learning training
- Batch processing of events
- Data augmentation
- Pipeline optimization

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/dataloader/cxx/*.cxx`` - Dataloader implementations
- ``src/AnalysisG/modules/dataloader/include/generators/*.h`` - Generator headers

Key Components
--------------

**Batch Generator**

Creates batches of data for training:

.. code-block:: cpp

   class batch_generator {
   public:
       // Load next batch
       std::vector<graph_t*> next_batch(size_t batch_size);
       
       // Shuffle data
       void shuffle();
       
       // Reset iterator
       void reset();
   };

**Data Pipeline**

Configures data loading pipeline:

- File reading
- Event parsing
- Graph construction
- Feature extraction
- Batching

Features
--------

**Multi-threaded Loading**

- Parallel file reading
- Background batch preparation
- Asynchronous I/O operations
- Thread-safe queuing

**Memory Management**

- Batch pooling
- Efficient memory reuse
- Automatic garbage collection
- Memory limit enforcement

**Preprocessing**

- On-the-fly feature computation
- Data normalization
- Augmentation support
- Filtering and selection

Usage Example
-------------

.. code-block:: cpp

   #include <generators/dataloader.h>
   
   // Create dataloader
   dataloader dl;
   dl.set_files(file_list);
   dl.set_batch_size(32);
   dl.set_num_workers(4);
   
   // Training loop
   while (dl.has_next()) {
       auto batch = dl.next_batch();
       // Train on batch
   }

Integration with Training
-------------------------

The dataloader integrates with:

- Model templates for training
- Optimizer for gradient updates
- Metric templates for evaluation
- Graph templates for data structure

Performance Optimization
------------------------

- Prefetching of next batches
- Memory-mapped file access
- Zero-copy operations where possible
- Efficient batch assembly

Configuration Options
---------------------

**Batch Size**

Control the number of samples per batch:

.. code-block:: cpp

   dl.batch_size = 32;

**Number of Workers**

Set parallel loading threads:

.. code-block:: cpp

   dl.num_workers = 4;

**Shuffle**

Enable data shuffling:

.. code-block:: cpp

   dl.shuffle = true;

**Drop Last**

Handle incomplete final batch:

.. code-block:: cpp

   dl.drop_last = false;

See Also
--------

* :doc:`model` - Model template using dataloaders
* :doc:`graph` - Graph template for data structure
* :doc:`io` - I/O operations
