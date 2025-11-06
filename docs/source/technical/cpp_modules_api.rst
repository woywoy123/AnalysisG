C++ Modules API Reference
==========================

This section documents the C++ implementations in the modules package. These are the high-performance backend implementations that power AnalysisG.

Overview
--------

The modules package contains 23 C++ modules that provide:

* Event and particle data structures
* Graph algorithms and representations  
* I/O operations for ROOT files
* Analysis orchestration
* Model training infrastructure
* Metrics and loss functions
* Optimization algorithms

All modules are in ``src/AnalysisG/modules/``.

Core Data Structures
--------------------

particle_template
~~~~~~~~~~~~~~~~~

**Location**: ``modules/particle/include/templates/particle_template.h``

The C++ particle template provides high-performance particle operations.

**Key Properties**:

.. code-block:: cpp

   class particle_template {
   public:
       // Four-momentum (cproperty provides getter/setter)
       cproperty<double, particle_template> px;
       cproperty<double, particle_template> py;
       cproperty<double, particle_template> pz;
       cproperty<double, particle_template> e;
       
       // Derived kinematics
       cproperty<double, particle_template> pt;
       cproperty<double, particle_template> eta;
       cproperty<double, particle_template> phi;
       cproperty<double, particle_template> mass;
       
       // Identification
       cproperty<int, particle_template> pdgid;
       cproperty<double, particle_template> charge;
       cproperty<std::string, particle_template> symbol;
       cproperty<std::string, particle_template> type;
       
       // Boolean properties for particle types
       cproperty<bool, particle_template> is_b;      // b-quark/jet
       cproperty<bool, particle_template> is_lep;    // lepton
       cproperty<bool, particle_template> is_nu;     // neutrino
       cproperty<bool, particle_template> is_add;    // additional particle
       cproperty<bool, particle_template> lep_decay; // from lepton decay
       
       // Decay chain
       cproperty<std::map<std::string, particle_template*>, particle_template> parents;
       cproperty<std::map<std::string, particle_template*>, particle_template> children;
       
       // Methods
       double DeltaR(particle_template* p);
       bool operator == (particle_template& p);
       template <typename g> g operator + (g& p);
       void operator += (particle_template* p);
   };

**Usage Pattern**:

.. code-block:: cpp

   particle_template* p = new particle_template();
   p->px = 100.0;  // MeV
   p->py = 50.0;
   p->pz = 200.0;
   p->e = 250.0;
   
   double pt_val = p->pt;   // Computed from px, py
   double eta_val = p->eta; // Computed from momentum
   
   // Check particle type
   if (p->is_lep) {
       std::cout << "Lepton found" << std::endl;
   }
   
   // Calculate angular separation
   double dr = p->DeltaR(other_particle);

event_template
~~~~~~~~~~~~~~

**Location**: ``modules/event/include/templates/event_template.h``

The C++ event template manages event-level data.

**Key Members**:

.. code-block:: cpp

   class event_template {
   public:
       // Event identification
       int index;
       double weight;
       std::string hash;
       std::string name;
       
       // ROOT tree information
       std::string tree;
       std::vector<std::string> trees;
       std::vector<std::string> branches;
       
       // Particle collections
       std::map<std::string, particle_template*> particles;
       std::map<std::string, particle_template*> tops;
       std::map<std::string, particle_template*> children;
       
       // Methods
       virtual bool selection();
       virtual void strategy();
       bool operator == (event_template& ev);
   };

**Inheritance Pattern**:

Users inherit from ``event_template`` in C++ headers, then wrap with Cython. The base class provides virtual methods ``selection()`` and ``strategy()`` that derived classes override.

graph_template
~~~~~~~~~~~~~~

**Location**: ``modules/graph/include/templates/graph_template.h``

Manages graph representations of events for GNN processing.

**Key Features**:

* Node and edge data structures
* Adjacency matrix computation
* Feature extraction
* Integration with PyTorch geometric

Analysis and I/O
----------------

analysis
~~~~~~~~

**Location**: ``modules/analysis/include/analysis/analysis.h``

The main analysis orchestrator that coordinates:

* Event processing loops
* Model training
* Metric evaluation
* Progress tracking
* Output management

**Key Methods**:

.. code-block:: cpp

   class analysis {
   public:
       void add_samples(std::string path, std::string label);
       void add_event_template(event_template* ev, std::string label);
       void add_graph_template(graph_template* gr, std::string label);
       void add_selection_template(selection_template* sel);
       void add_model(/* model params */);
       
       void start();  // Begin processing
       
       std::map<std::string, std::vector<float>> progress();
       std::map<std::string, bool> is_complete();
   };

io Module
~~~~~~~~~

**Location**: ``modules/io/include/io/*.h``

Handles ROOT file I/O operations:

* Reading ROOT TTrees
* Branch parsing
* Event deserialization  
* Efficient memory management

**Key Classes**:

* ``root_reader``: Reads ROOT files
* ``tree_reader``: Handles TTree access
* ``branch_reader``: Reads specific branches

Graph Algorithms
----------------

graph Module  
~~~~~~~~~~~~

**Location**: ``modules/graph/``

Provides graph construction and algorithms:

* Edge list generation
* Adjacency matrix computation
* Graph traversal
* Connected components
* Topological operations

**Implementation**: Uses efficient C++ STL containers (``std::vector``, ``std::map``, ``std::unordered_map``) for performance.

Machine Learning Infrastructure
--------------------------------

model Module
~~~~~~~~~~~~

**Location**: ``modules/model/``

Backend for model training:

* Model state management
* Batch processing
* Gradient computation interface
* Checkpoint saving/loading

optimizer Module
~~~~~~~~~~~~~~~~

**Location**: ``modules/optimizer/``

Optimization algorithms:

* Adam optimizer
* SGD variants
* Learning rate scheduling
* Parameter updates

lossfx Module
~~~~~~~~~~~~~

**Location**: ``modules/lossfx/``

Loss function implementations:

* Cross-entropy
* MSE
* Custom physics-motivated losses

Utilities
---------

tools Module
~~~~~~~~~~~~

**Location**: ``modules/tools/include/tools/tools.h``

Common utility functions:

* String encoding/decoding
* Vector operations
* Mathematical utilities
* Type conversions

structs Module
~~~~~~~~~~~~~~

**Location**: ``modules/structs/``

Core data structures:

* ``particle_t``: Particle data struct
* ``property_t``: Property containers
* ``element_t``: Generic elements

typecasting Module
~~~~~~~~~~~~~~~~~~

**Location**: ``modules/typecasting/``

Type conversion utilities between:

* Python types
* C++ types
* ROOT types

Advanced Features
-----------------

nusol Module
~~~~~~~~~~~~

**Location**: ``modules/nusol/``

Neutrino reconstruction algorithms:

* Missing momentum calculation
* Neutrino four-momentum estimation
* Multiple solution handling
* Constraint optimization

**Algorithms**:

* Analytical solutions for W→lν
* Numerical optimization for top reconstruction
* Multiple hypothesis testing

sampletracer Module
~~~~~~~~~~~~~~~~~~~

**Location**: ``modules/sampletracer/``

Tracks sample metadata:

* Cross-sections
* Sum of weights
* Luminosity
* Sample provenance

meta Module
~~~~~~~~~~~

**Location**: ``modules/meta/``

Metadata management:

* Dataset information
* Processing history
* Configuration tracking

Performance Considerations
--------------------------

The C++ modules are designed for maximum performance:

1. **Memory Management**: Uses RAII and smart pointers where appropriate
2. **Cache Efficiency**: Data structures optimized for cache locality
3. **Parallelization**: Thread-safe where needed, OpenMP support
4. **Vectorization**: Compiler auto-vectorization friendly code
5. **Move Semantics**: Modern C++ features for efficient transfers

Building the C++ Code
----------------------

The C++ code is built as part of the AnalysisG installation:

.. code-block:: bash

   pip install -e .

This compiles the C++ modules and creates Cython wrappers automatically.

**Dependencies**:

* C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
* ROOT 6.x (for I/O)
* PyTorch C++ API (for models)

See Also
--------

* :doc:`cuda_api`: CUDA implementations
* :doc:`../modules/overview`: Module organization
* :doc:`overview`: Technical overview
