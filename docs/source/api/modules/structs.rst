Structs Module (C++)
====================

The Structs module provides core data structure definitions.

Overview
--------

Located in ``src/AnalysisG/modules/structs/``, this module implements fundamental 
data structures in C++:

- Event data structures
- Particle data structures
- Graph data structures
- Metadata structures
- Property structures

Purpose
-------

The structs module defines:

- Common data layouts
- Type-safe structures
- Memory-efficient representations
- Interoperability between components

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/structs/cxx/*.cxx`` - Structure implementations
- ``src/AnalysisG/modules/structs/include/structs/*.h`` - Structure headers

**Python Binding**

- ``src/AnalysisG/core/structs.pyx`` - Cython wrapper
- ``src/AnalysisG/core/structs.pxd`` - Cython declarations

Key Structures
--------------

**particle_t**

Particle data:

.. code-block:: cpp

   struct particle_t {
       // Kinematics
       double pt, eta, phi, e;
       double px, py, pz;
       double mass;
       
       // Identification
       int pdgid;
       double charge;
       std::string symbol;
       
       // Relationships
       std::vector<int> parent_indices;
       std::vector<int> children_indices;
       
       // Hash
       std::string hash;
   };

**event_t**

Event data:

.. code-block:: cpp

   struct event_t {
       // Event identification
       long event_number;
       int run_number;
       double weight;
       
       // Particles
       std::map<std::string, std::vector<particle_t>> particles;
       
       // Event-level quantities
       double met, met_phi;
       double ht;
       int n_jets, n_bjets, n_leptons;
       
       // Metadata
       std::string filename;
       std::string tree_name;
       std::string hash;
   };

**graph_t**

Graph data for GNNs:

.. code-block:: cpp

   struct graph_t {
       // Nodes (particles)
       std::vector<int> node_indices;
       std::map<std::string, std::vector<double>> node_features;
       
       // Edges
       std::vector<std::pair<int, int>> edge_index;
       std::map<std::string, std::vector<double>> edge_features;
       
       // Graph-level features
       std::map<std::string, double> graph_features;
       
       // Labels
       int label;
       std::vector<double> label_vector;
   };

**meta_t**

Metadata structure:

.. code-block:: cpp

   struct meta_t {
       // Dataset info
       int dsid;
       std::string dataset_name;
       
       // Cross-sections
       double cross_section;
       double filter_efficiency;
       double k_factor;
       
       // Event counts
       long total_events;
       double sum_weights;
       
       // Generator
       std::string generator;
       std::string campaign;
   };

**element_t**

ROOT data element:

.. code-block:: cpp

   struct element_t {
       // Data pointers
       void* data;
       std::string type;
       size_t size;
       
       // Branch information
       std::string tree_name;
       std::string branch_name;
       std::string leaf_name;
       
       // Access methods
       template<typename T>
       T* get() { return static_cast<T*>(data); }
   };

**data_t**

Generic data container:

.. code-block:: cpp

   struct data_t {
       std::map<std::string, element_t> elements;
       long entry_number;
       std::string source_file;
       
       // Access
       element_t& operator[](std::string key) {
           return elements[key];
       }
   };

Property Structures
-------------------

**cproperty**

Template property with automatic getter/setter:

.. code-block:: cpp

   template<typename T, typename C>
   struct cproperty {
       T value;
       C* owner;
       
       void (*setter)(T*, C*);
       void (*getter)(T*, C*);
       
       operator T() const {
           T result;
           if (getter) getter(&result, owner);
           return result;
       }
       
       cproperty& operator=(const T& val) {
           if (setter) setter(&val, owner);
           return *this;
       }
   };

Usage:

.. code-block:: cpp

   class MyClass {
   public:
       cproperty<double, MyClass> value;
       
       static void set_value(double* v, MyClass* obj) {
           obj->internal_value = *v;
           // Additional logic
       }
       
       static void get_value(double* v, MyClass* obj) {
           *v = obj->internal_value;
           // Additional logic
       }
       
   private:
       double internal_value;
   };

Selection Structures
--------------------

**selection_t**

Selection results:

.. code-block:: cpp

   struct selection_t {
       std::string name;
       bool passed;
       
       // Cut results
       std::map<std::string, bool> cut_results;
       
       // Statistics
       long events_processed;
       long events_passed;
       double sum_weights_passed;
   };

Metric Structures
-----------------

**metric_t**

Metric data:

.. code-block:: cpp

   struct metric_t {
       std::string name;
       std::map<std::string, std::vector<double>> values;
       std::map<std::string, std::string> run_names;
   };

Model Structures
----------------

**model_state_t**

Model state:

.. code-block:: cpp

   struct model_state_t {
       std::string model_name;
       int epoch;
       std::map<std::string, torch::Tensor> parameters;
       std::map<std::string, double> metrics;
   };

Memory Layout
-------------

Structures are designed for:

- Cache-friendly access patterns
- Minimal padding
- Efficient serialization
- Standard layout compatibility

Serialization
-------------

Structures support serialization:

.. code-block:: cpp

   // To JSON
   nlohmann::json to_json(const particle_t& p);
   
   // From JSON
   particle_t from_json(const nlohmann::json& j);
   
   // To binary
   void serialize(std::ostream& os, const particle_t& p);
   
   // From binary
   particle_t deserialize(std::istream& is);

Type Safety
-----------

Structures provide type safety:

.. code-block:: cpp

   // Compile-time type checking
   particle_t p;
   p.pt = 100.0;           // OK
   // p.pt = "invalid";    // Compile error
   
   // Runtime type checking
   element_t elem;
   if (elem.type == "double") {
       auto val = elem.get<double>();
   }

Integration
-----------

Structures integrate with:

- ROOT I/O for persistence
- Cython for Python access
- LibTorch for tensor operations
- Standard algorithms

See Also
--------

* :doc:`../core/structs` - Python Structs wrapper
* :doc:`event` - Event template using structures
* :doc:`particle` - Particle template using structures
* :doc:`graph` - Graph template using structures
