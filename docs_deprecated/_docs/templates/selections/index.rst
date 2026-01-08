Physics Selections
=================

The selections system in AnalysisG allows you to define physics selection regions and criteria for analysis. Selection templates are found in the ``src/AnalysisG/selections/`` directory.

.. toctree::
   :maxdepth: 1
   
   analysis/index
   neutrino/index
   templates/index

Selection Model
--------------

Selections in AnalysisG are implemented through the ``selection_template`` base class, which provides the interface for all selection types. Specific selections inherit from this template.

.. code-block:: cpp

   // Base selection template interface
   class selection_template: public tools
   {
       public:
           virtual selection_template* clone() = 0;
           virtual bool selection(event_template* ev) = 0;
           virtual bool strategy(event_template* ev) = 0;
           virtual void merge(selection_template* sl) = 0;
           // ...
   };

Selection Flow
-------------

Selections are applied to events in a structured manner:

1. Selections are registered with the analysis
2. Events are processed and passed to selections
3. Selection criteria are evaluated
4. Events passing selections are stored or further processed
5. Statistical information about selections is accumulated

Key Selection Types
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Selection
     - Description
   * - ``regions``
     - Physics analysis regions (signal and control)
   * - ``validation``
     - Validation regions for performance checks
   * - ``benchmark``
     - Benchmarking selections for performance studies

Analysis Regions
---------------

A key concept in physics analysis is the definition of signal and control regions. In AnalysisG, these are typically defined in a ``regions`` selection:

.. code-block:: cpp

   struct package_t {
       regions_t CRttbarCO2l_CO; 
       regions_t CRttbarCO2l_CO_2b; 
       regions_t VRttZ3l; 
       regions_t SR4b; 
       regions_t SR2b; 
       regions_t SR3b; 
       // ...
   };

Creating Custom Selections
-------------------------

To create a custom selection:

1. Inherit from ``selection_template``
2. Implement the required virtual methods
3. Define selection criteria in the ``selection`` method
4. Implement the ``strategy`` method for additional logic
5. Implement the ``merge`` method to combine selections

Example Selection Template
------------------------

.. code-block:: cpp

   class custom_selection: public selection_template
   {
       public:
           custom_selection() { this->name = "custom_selection"; }
           ~custom_selection() {}
           selection_template* clone() { return (selection_template*)new custom_selection(); }
           
           bool selection(event_template* ev) {
               // Implement selection criteria
               return true;
           }
           
           bool strategy(event_template* ev) {
               // Additional selection logic
               return true;
           }
           
           void merge(selection_template* sl) {
               // Handle merging selections
           }
   };