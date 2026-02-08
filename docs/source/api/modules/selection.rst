Selection Module (C++)
======================

The Selection module provides the C++ implementation of selection templates for event filtering.

Overview
--------

Located in ``src/AnalysisG/modules/selection/``, this module implements selection 
template functionality in C++ for defining event selection criteria:

- Event filtering and cut-based selections
- Selection logic implementation
- Event counting and weight tracking
- Serialization for analysis preservation

C++ Class: selection_template
------------------------------

Header Location
~~~~~~~~~~~~~~~

``src/AnalysisG/modules/selection/include/templates/selection_template.h``

Key Properties
~~~~~~~~~~~~~~

The ``selection_template`` class uses ``cproperty`` template accessors:

.. code-block:: cpp

   cproperty<std::string, selection_template> name;      // Selection name
   cproperty<std::string, selection_template> hash;      // Unique hash
   cproperty<bool, selection_template> passed;           // Event passed selection

Key Methods
~~~~~~~~~~~

**Template Management**

.. cpp:function:: virtual selection_template* clone()

   Create a copy of the selection template.

.. cpp:function:: virtual bool Selection(event_template* ev)

   Evaluate selection criteria for an event.

   :param ev: Event template to evaluate
   :return: True if event passes selection, False otherwise

   **Note**: This is a virtual method that should be overridden in derived classes
   to implement custom selection logic.

.. cpp:function:: virtual void Strategy(event_template* ev)

   Perform additional calculations on selected events.

   :param ev: Event template that passed selection

   This method is called after ``Selection()`` returns true and can be used to:
   
   - Calculate derived quantities
   - Fill histograms
   - Perform truth matching
   - Compute selection-specific variables

**Serialization**

.. cpp:function:: void dump(std::string path, std::string name = "")

   Save selection results to file.

   :param path: Directory path for output
   :param name: Output filename (defaults to selection name)

.. cpp:function:: selection_template* load(std::string path, std::string name = "")

   Load selection results from file.

   :param path: Directory path for input
   :param name: Input filename (defaults to selection name)
   :return: Loaded selection template or nullptr if failed

**Weight Management**

.. cpp:function:: std::map<std::string, std::map<std::string, float>> passed_weights()

   Get map of event weights for events that passed selection.

   :return: Map of [filename -> [event_hash -> weight]]

.. cpp:function:: std::vector<std::pair<std::string, float>> HashToWeightFile(std::vector<std::string> hashes)

   Convert event hashes to (filename, weight) pairs.

   :param hashes: Vector of event hash strings
   :return: Vector of (filename, weight) tuples for each hash

Member Variables
~~~~~~~~~~~~~~~~

.. cpp:member:: std::map<std::string, std::map<std::string, float>> m_passed_weights

   Storage for weights of events passing selection, organized by file and event hash.

.. cpp:member:: std::map<std::string, meta_t> m_matched_meta

   Metadata associated with selected events.

.. cpp:member:: selection_t data

   Selection data structure containing results and statistics.

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/selection/cxx/selection_template.cxx`` - Main implementation
- ``src/AnalysisG/modules/selection/cxx/properties.cxx`` - Property implementations
- ``src/AnalysisG/modules/selection/cxx/serialization.cxx`` - Save/load functionality

**Python Binding**

- ``src/AnalysisG/core/selection_template.pyx`` - Cython wrapper
- ``src/AnalysisG/core/selection_template.pxd`` - Cython declarations

Usage from C++
--------------

**Defining a Selection**

.. code-block:: cpp

   #include <templates/selection_template.h>
   #include <templates/event_template.h>
   
   class TTbarSelection : public selection_template {
   public:
       bool Selection(event_template* ev) override {
           // Require at least 4 jets
           if (ev->n_jets < 4) return false;
           
           // Require at least 1 lepton
           if (ev->n_leptons < 1) return false;
           
           // Require MET > 20 GeV
           if (ev->met < 20000) return false;
           
           return true;
       }
       
       void Strategy(event_template* ev) override {
           // Calculate HT for selected events
           double ht = 0;
           for (auto& jet : ev->jets) {
               ht += jet->pt;
           }
           ev->ht = ht;
           
           // Perform additional analysis
           // ...
       }
   };

**Using the Selection**

.. code-block:: cpp

   // Create selection instance
   TTbarSelection* sel = new TTbarSelection();
   sel->name = "TTbar_Selection";
   
   // Evaluate for an event
   event_template* event = /* ... */;
   bool passes = sel->Selection(event);
   
   if (passes) {
       sel->Strategy(event);
       // Event passed, continue analysis
   }
   
   // Save results
   sel->dump("./output", "ttbar_selection");
   
   // Load results later
   TTbarSelection* loaded = new TTbarSelection();
   loaded = (TTbarSelection*)loaded->load("./output", "ttbar_selection");

Integration with Python
-----------------------

The C++ selection_template is wrapped in Python as ``SelectionTemplate``:

.. code-block:: python

   from AnalysisG.core.selection_template import SelectionTemplate
   
   class TTbarSelection(SelectionTemplate):
       def __init__(self):
           super().__init__()
       
       # Selection logic is implemented in C++
       # Python wrapper provides access to results

   # Use in analysis
   from AnalysisG import Analysis
   
   sel = TTbarSelection()
   ana = Analysis()
   ana.AddSelection(sel)
   ana.Start()
   
   # Access results
   weights = sel.PassedWeights
   metadata = sel.GetMetaData

Selection Patterns
------------------

**Cut-based Selection**

Traditional HEP analysis cuts:

.. code-block:: cpp

   bool Selection(event_template* ev) override {
       // Trigger requirement
       if (!ev->trigger_passed) return false;
       
       // Object quality cuts
       if (ev->n_good_jets < 6) return false;
       if (ev->n_good_leptons < 1) return false;
       
       // Kinematic cuts
       if (ev->met < 20000) return false;  // MET > 20 GeV
       if (ev->leading_jet_pt < 25000) return false;  // pT > 25 GeV
       
       // Topology cuts
       if (ev->n_bjets < 2) return false;
       
       return true;
   }

**Pre-selection for ML**

Loose selection before ML algorithms:

.. code-block:: cpp

   bool Selection(event_template* ev) override {
       // Minimal cuts to ensure event quality
       if (ev->n_jets < 4) return false;
       if (ev->leading_jet_pt < 20000) return false;
       
       // Save all events passing minimal cuts for ML training
       return true;
   }

**Multi-region Selection**

Define multiple signal/control regions:

.. code-block:: cpp

   class MultiRegionSelection : public selection_template {
   public:
       std::string region;
       
       bool Selection(event_template* ev) override {
           // Base selection
           if (ev->n_jets < 4) return false;
           
           // Determine region
           if (ev->n_leptons == 0) {
               region = "0L";  // Zero lepton
           } else if (ev->n_leptons == 1) {
               region = "1L";  // Single lepton
           } else {
               region = "2L";  // Dilepton
           }
           
           // Region-specific cuts
           if (region == "0L" && ev->met < 200000) return false;
           if (region == "1L" && ev->mt < 100000) return false;
           
           return true;
       }
   };

Statistics and Reporting
-------------------------

The selection template automatically tracks:

- Number of events processed
- Number of events passing selection
- Cumulative event weights
- Per-file statistics
- Metadata associations

This information is available for:

- Cutflow tables
- Efficiency calculations
- Systematic uncertainty propagation
- Result validation

Example Implementations
-----------------------

Example selection templates are provided in ``src/AnalysisG/selections/``:

- ``selections/example/met/`` - MET-based selection
- ``selections/example/*/`` - Various analysis-specific selections

These serve as templates for implementing custom selections.

See Also
--------

* :doc:`../core/templates` - Python SelectionTemplate documentation
* :doc:`event` - Event template C++ implementation
* :doc:`analysis` - Analysis engine using selections
