Templates Module
================

The Templates module provides boilerplate code and examples for creating
custom AnalysisG components.

For complete API reference, see the Doxygen-generated HTML documentation in ``doxygen-docs/html/``.

Template Types
--------------

Event Templates
~~~~~~~~~~~~~~~

Templates for custom event types.

**Location**: ``src/AnalysisG/templates/events/``

Includes:
* Event class structure
* Particle registration
* Event compilation logic
* Custom property definitions

Particle Templates
~~~~~~~~~~~~~~~~~~

Templates for custom particle types.

**Location**: ``src/AnalysisG/templates/particles/``

Includes:
* Particle class structure
* Property definitions
* Kinematic calculations
* Particle identification

Common properties:
* Four-momentum components
* Charge, mass, PDG ID
* Isolation variables
* Quality flags

Graph Templates
~~~~~~~~~~~~~~~

Templates for graph construction.

**Location**: ``src/AnalysisG/templates/graphs/``

Customization points:
* Node feature selection
* Edge connectivity rules
* Feature normalization
* Graph-level aggregation

Model Templates
~~~~~~~~~~~~~~~

Templates for neural network models.

**Location**: ``src/AnalysisG/templates/model/``

Architecture components:
* Input processing layers
* Hidden layers
* Output layers
* Activation functions

Metric Templates
~~~~~~~~~~~~~~~~

Templates for custom metrics.

**Location**: ``src/AnalysisG/templates/metrics/``

Metric types:
* Classification metrics
* Regression metrics
* Physics-specific metrics
* Custom loss functions

Selection Templates
~~~~~~~~~~~~~~~~~~~

Templates for event selections.

**Location**: ``src/AnalysisG/templates/selections/``

Selection features:
* Sequential cuts
* Event categorization
* Weight handling
* Histogram filling

Using Templates
---------------

Typical Workflow
~~~~~~~~~~~~~~~~

1. Copy template directory:

.. code-block:: bash

   cp -r templates/events/eventtemplate myanalysis/events/myevent

2. Customize names:

.. code-block:: cpp

   // Change from:
   class EventTemplate : public event_template
   // To:
   class MyEvent : public event_template

3. Implement logic:

.. code-block:: cpp

   void MyEvent::CompileEvent() {
       // Custom event processing
       compute_custom_variables();
       apply_corrections();
   }

4. Build and test:

.. code-block:: bash

   mkdir build && cd build
   cmake ..
   make

Template Structure
------------------

Each template includes:

* Header file (``.h``) - Class declaration
* Implementation file (``.cxx``) - Method implementations
* CMakeLists.txt - Build configuration
* Documentation (``.md``) - Usage instructions

Best Practices
--------------

When using templates:

* **Keep Original**: Don't modify templates directly
* **Rename Properly**: Use consistent naming
* **Document Changes**: Add comments
* **Follow Conventions**: Maintain coding style
* **Test Thoroughly**: Validate implementations

Example Customizations
----------------------

Custom Event
~~~~~~~~~~~~

.. code-block:: cpp

   class TopPairEvent : public event_template {
       public:
           void CompileEvent() override {
               find_top_candidates();
               compute_masses();
               categorize_event();
           }
           
       private:
           void find_top_candidates();
           void compute_masses();
           void categorize_event();
           
           double m_ttbar;
           int n_bjets;
   };

Custom Metric
~~~~~~~~~~~~~

.. code-block:: cpp

   class TopMassAccuracy : public metric_template {
       public:
           void event() override {
               double true_mass = get_true_top_mass();
               double pred_mass = get_predicted_top_mass();
               double error = std::abs(true_mass - pred_mass);
               
               total_error += error;
               n_events++;
           }
           
           void end() override {
               mean_error = total_error / n_events;
           }
           
       private:
           double total_error = 0.0;
           int n_events = 0;
           double mean_error = 0.0;
   };

Deprecated Documentation
------------------------

Original template documentation has been moved to:

* ``src/deprecated/eventtemplate.md``
* ``src/deprecated/modeltemplate.md``
* ``src/deprecated/selectiontemplate.md``
* ``src/deprecated/graphtemplate.md``
* ``src/deprecated/particletemplate.md``
