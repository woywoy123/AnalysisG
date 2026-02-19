Container Module (C++)
======================

The Container module provides data container structures for AnalysisG.

Overview
--------

Located in ``src/AnalysisG/modules/container/``, this module implements C++ container 
classes for managing collections of analysis objects:

- Event collections
- Particle collections
- Graph data structures
- Efficient memory management
- Iterator support

Purpose
-------

The container module provides specialized data structures optimized for:

- Storing large numbers of events and particles
- Fast iteration and access
- Memory-efficient storage
- Type-safe collections
- Integration with ROOT containers

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/container/cxx/*.cxx`` - Container implementations
- ``src/AnalysisG/modules/container/include/container/*.h`` - Container headers

Key Container Types
-------------------

**Event Containers**

Specialized containers for managing event collections:

- Efficient storage of event_template instances
- Support for event indexing and lookup
- Memory pooling for performance

**Particle Containers**

Containers for particle collections:

- Vector-based storage for fast iteration
- Map-based storage for keyed access
- Support for parent-child relationships

**Graph Containers**

Containers for graph data:

- Adjacency list representations
- Edge and node feature storage
- Integration with PyTorch Geometric

Usage Patterns
--------------

Containers provide standard C++ container interfaces:

.. code-block:: cpp

   // Create container
   event_container events;
   
   // Add elements
   events.push_back(event_ptr);
   
   // Iterate
   for (auto& evt : events) {
       // Process event
   }
   
   // Access by index
   auto& evt = events[i];

Memory Management
-----------------

Containers handle:

- Automatic memory allocation
- Object lifetime management
- Efficient resizing strategies
- Cache-friendly data layout

Performance Considerations
--------------------------

- Contiguous memory layout for cache efficiency
- Pre-allocation to avoid reallocation overhead
- Move semantics for efficient transfers
- RAII for automatic cleanup

See Also
--------

* :doc:`structs` - Data structure definitions
* :doc:`event` - Event template
* :doc:`particle` - Particle template
