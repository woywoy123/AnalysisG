Events Module
=============

The Events module provides concrete implementations of event processing for
various physics analyses.

For complete API reference, see the Doxygen-generated HTML documentation in ``doxygen-docs/html/``.

Event Implementations
---------------------

BSM Four-Top Events
~~~~~~~~~~~~~~~~~~~

Event processing for Beyond Standard Model four-top quark production.

**Location**: ``src/AnalysisG/events/bsm_4tops/``

Experimental MC20 Events
~~~~~~~~~~~~~~~~~~~~~~~~~

Event processing for MC20 experimental data format.

**Location**: ``src/AnalysisG/events/exp_mc20/``

GNN Inference Events
~~~~~~~~~~~~~~~~~~~~

Event structure optimized for Graph Neural Network inference.

**Location**: ``src/AnalysisG/events/gnn/``

SSML MC20 Events
~~~~~~~~~~~~~~~~

Event processing for Same-Sign Multi-Lepton analysis.

**Location**: ``src/AnalysisG/events/ssml_mc20/``

Usage Pattern
-------------

All event implementations follow a common interface inherited from
``event_template``. See the Core module documentation for base class details.

Example
~~~~~~~

.. code-block:: cpp

   // Create event instance
   auto* evt = new EventType();
   
   // Build event from raw data
   evt->build(element);
   
   // Compile event
   evt->CompileEvent();
   
   // Access particles
   auto particles = evt->Particles;
