Event Module (C++)
==================

The Event module provides the C++ implementation of event templates.

Overview
--------

Located in ``src/AnalysisG/modules/event/``, this module implements the core event 
template functionality in C++:

- Event structure definition
- ROOT data mapping
- Particle registration and management
- Event building and compilation

C++ Class: event_template
--------------------------

Header Location
~~~~~~~~~~~~~~~

``src/AnalysisG/modules/event/include/templates/event_template.h``

Key Properties
~~~~~~~~~~~~~~

The ``event_template`` class uses ``cproperty`` template accessors:

**Tree Configuration**

.. code-block:: cpp

   cproperty<std::vector<std::string>, event_template> trees;
   cproperty<std::vector<std::string>, event_template> branches;
   cproperty<std::vector<std::string>, event_template> leaves;
   cproperty<std::string, event_template> tree;

**Event Metadata**

.. code-block:: cpp

   cproperty<std::string, event_template> name;
   cproperty<std::string, event_template> hash;
   cproperty<double, event_template> weight;
   cproperty<long, event_template> index;

Key Methods
~~~~~~~~~~~

**Template Management**

.. cpp:function:: virtual event_template* clone()

   Create a copy of the event template.

.. cpp:function:: virtual void build(element_t* el)

   Build event from ROOT data element.

.. cpp:function:: virtual void CompileEvent()

   Compile and finalize event structure.

**Particle Registration**

.. cpp:function:: template <typename G> void register_particle(std::map<std::string, G*>* object)

   Register a particle type with the event template.

   :param object: Map to store particle instances
   :template G: Particle template type derived from particle_template

.. cpp:function:: template <typename G> void deregister_particle(std::map<std::string, G*>* object)

   Deregister and cleanup particle instances.

   :param object: Map containing particle instances to cleanup

**Event Building**

.. cpp:function:: std::map<std::string, event_template*> build_event(std::map<std::string, data_t*>* evnt)

   Build event objects from ROOT data.

   :param evnt: Map of ROOT data elements
   :return: Map of built event templates

**Neutrino Reconstruction**

.. cpp:function:: std::vector<particle_template*> multi_neutrino( \
                  std::vector<particle_template*>* targets, \
                  double phi, double met, \
                  double mt = 172.68 * 1000, \
                  double mw = 80.385 * 1000, \
                  double violation = 1e-4, \
                  double limit = 0.1)

   Perform multi-neutrino reconstruction.

   :param targets: Target particles (typically leptons)
   :param phi: MET phi angle
   :param met: Missing transverse energy
   :param mt: Top quark mass (default: 172.68 GeV)
   :param mw: W boson mass (default: 80.385 GeV)
   :param violation: Convergence tolerance
   :param limit: Constraint limit
   :return: Vector of reconstructed neutrino particles

Member Variables
~~~~~~~~~~~~~~~~

.. cpp:member:: event_t data

   Event data structure containing parsed ROOT information.

.. cpp:member:: meta* meta_data

   Pointer to metadata associated with this event.

.. cpp:member:: std::string filename

   Source filename for this event.

.. cpp:member:: std::map<std::string, std::string> m_trees

   Mapping of tree names to ROOT tree paths.

.. cpp:member:: std::map<std::string, std::string> m_branches

   Mapping of branch names to ROOT branch paths.

.. cpp:member:: std::map<std::string, std::string> m_leaves

   Mapping of leaf names to ROOT leaf paths.

Property Setters and Getters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``cproperty`` template provides automatic setter/getter generation:

.. code-block:: cpp

   // Setters
   static void set_trees(std::vector<std::string>*, event_template*);
   static void set_branches(std::vector<std::string>*, event_template*);
   static void set_name(std::string*, event_template*);
   static void set_hash(std::string*, event_template*);
   static void set_tree(std::string*, event_template*);
   static void set_weight(double*, event_template*);
   static void set_index(long*, event_template*);
   
   // Getters
   static void get_leaves(std::vector<std::string>*, event_template*);
   static void get_hash(std::string*, event_template*);
   static void get_tree(std::string*, event_template*);

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/event/cxx/event_template.cxx`` - Main implementation
- ``src/AnalysisG/modules/event/cxx/properties.cxx`` - Property implementations

**Python Binding**

- ``src/AnalysisG/core/event_template.pyx`` - Cython wrapper
- ``src/AnalysisG/core/event_template.pxd`` - Cython declarations

Usage from C++
--------------

.. code-block:: cpp

   #include <templates/event_template.h>
   
   // Create event template
   event_template* evt = new event_template();
   
   // Configure ROOT access
   evt->tree = "nominal";
   evt->branches = {"jets_pt", "jets_eta", "met_met"};
   
   // Build event from data
   evt->build(data_element);
   evt->CompileEvent();
   
   // Access event properties
   double weight = evt->weight;
   long index = evt->index;

Integration with Python
-----------------------

The C++ event_template is wrapped in Python as ``EventTemplate``:

.. code-block:: python

   from AnalysisG.core.event_template import EventTemplate
   
   class MyEvent(EventTemplate):
       def __init__(self):
           super().__init__()
           self.Tree = "nominal"
           self.Branches = ["jets_pt", "jets_eta"]

See Also
--------

* :doc:`../core/templates` - Python EventTemplate documentation
* :doc:`particle` - Particle template C++ implementation
* :doc:`graph` - Graph template C++ implementation
* :doc:`analysis` - Analysis engine using event templates
