Core Module
===========

Overview
--------

The Core module provides the fundamental building blocks and base templates
for the entire AnalysisG framework. All higher-level components inherit from
these base classes.

Purpose
-------

The Core module serves as the foundation layer, providing:

* Base template classes for extension
* Common functionality shared across the framework
* Interface definitions for framework components
* Utility classes for common operations

Template Classes
----------------

The core module defines several base template classes that serve as interfaces
for custom implementations:

event_template
~~~~~~~~~~~~~~

Base class for all event processing implementations.

**Key Methods:**

* ``build()`` - Build event from raw data
* ``CompileEvent()`` - Process and compile event data
* ``clone()`` - Create a copy of the event

**Key Properties:**

* ``trees`` - Tree names for data access
* ``branches`` - Branch names for data access
* ``weight`` - Event weight
* ``index`` - Event index

particle_template
~~~~~~~~~~~~~~~~~

Base class for particle representations.

**Key Methods:**

* Kinematic accessors (pt, eta, phi, mass)
* Four-vector operations
* Particle identification

graph_template
~~~~~~~~~~~~~~

Base class for graph construction from events.

**Key Methods:**

* ``build()`` - Construct graph from event
* Node/edge feature extraction
* Graph property computation

model_template
~~~~~~~~~~~~~~

Base class for neural network models.

**Key Methods:**

* ``forward()`` - Forward pass computation
* ``clone()`` - Clone model for distributed training

metric_template
~~~~~~~~~~~~~~~

Base class for evaluation metrics.

**Key Methods:**

* ``define_metric()`` - Initialize metric
* ``define_variables()`` - Setup data collection
* ``event()`` - Per-event computation
* ``batch()`` - Per-batch aggregation
* ``end()`` - Final computation

selection_template
~~~~~~~~~~~~~~~~~~

Base class for event selection criteria.

**Key Methods:**

* Selection logic implementation
* Cut flow management
* Efficiency tracking

Utility Classes
---------------

tools
~~~~~

Common utility functions and helper methods.

meta
~~~~

Metadata management for datasets and runs.

structs
~~~~~~~

Common data structures and type definitions.

io
~~

Input/output operations for data files.

notification
~~~~~~~~~~~~

Notification and messaging system.

plotting
~~~~~~~~

Plotting and visualization utilities.

roc
~~~

ROC curve analysis and computation.

lossfx
~~~~~~

Loss function implementations.

analysis
~~~~~~~~

Analysis tools and utilities.

Design Philosophy
-----------------

The Core module follows these principles:

Extensibility
~~~~~~~~~~~~~

Template-based architecture allows easy extension through inheritance:

.. code-block:: cpp

   class MyEvent : public event_template {
       void CompileEvent() override {
           // Custom implementation
       }
   };

Reusability
~~~~~~~~~~~

Common functionality is implemented in base classes to avoid code duplication.

Performance
~~~~~~~~~~~

Efficient C++ implementation with minimal overhead.

Flexibility
~~~~~~~~~~~

Virtual methods allow customization at any level.

Integration
~~~~~~~~~~~

Seamless integration with Python via Cython bindings.

Usage Patterns
--------------

Creating Custom Events
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   class MyAnalysisEvent : public event_template {
   public:
       void CompileEvent() override {
           // Custom event processing
           compute_custom_variables();
           classify_event();
       }
   
   private:
       void compute_custom_variables();
       void classify_event();
   };

Creating Custom Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   class MyMetric : public metric_template {
   public:
       void event() override {
           // Per-event metric computation
       }
       
       void batch() override {
           // Batch-level aggregation
       }
       
       void end() override {
           // Final metric computation
       }
   };

See Also
--------

* :doc:`../api/core` - Detailed API documentation
* :doc:`templates` - Code templates and examples
* :doc:`events` - Event implementations
* :doc:`models` - Model implementations
