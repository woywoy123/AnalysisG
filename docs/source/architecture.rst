Architecture Overview
=====================

AnalysisG is a multi-layered framework combining C++, Python, and CUDA for high-performance particle physics analysis.

System Architecture
-------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                      Python Interface                        │
   │  (User Scripts, Jupyter Notebooks, Analysis Workflows)      │
   └───────────────────┬─────────────────────────────────────────┘
                       │
   ┌───────────────────┴─────────────────────────────────────────┐
   │                    Cython Wrappers                           │
   │  (src/AnalysisG/core/*.pyx - Python/C++ Bridge)            │
   │   • analysis.pyx - Analysis orchestration                   │
   │   • event_template.pyx - Event handling                     │
   │   • particle_template.pyx - Particle objects                │
   │   • graph_template.pyx - Graph construction                 │
   │   • selection_template.pyx - Event filtering                │
   │   • io.pyx - ROOT file access                               │
   └───────────────────┬─────────────────────────────────────────┘
                       │
   ┌───────────────────┴─────────────────────────────────────────┐
   │                   C++ Core Engine                            │
   │  (src/AnalysisG/modules/ - High-performance backend)       │
   │                                                              │
   │  ┌──────────────────────────────────────────────────────┐  │
   │  │ Template System (Virtual base classes)               │  │
   │  │  • event_template - Event structure                  │  │
   │  │  • particle_template - Particle physics              │  │
   │  │  • graph_template - GNN graphs                       │  │
   │  │  • selection_template - Event selection              │  │
   │  │  • model_template - ML models                        │  │
   │  │  • metric_template - Evaluation metrics              │  │
   │  └──────────────────────────────────────────────────────┘  │
   │                                                              │
   │  ┌──────────────────────────────────────────────────────┐  │
   │  │ Analysis Engine                                       │  │
   │  │  • Multi-threaded execution                          │  │
   │  │  • Progress tracking                                 │  │
   │  │  • Sample management                                 │  │
   │  │  • Template orchestration                            │  │
   │  └──────────────────────────────────────────────────────┘  │
   │                                                              │
   │  ┌──────────────────────────────────────────────────────┐  │
   │  │ Specialized Modules                                   │  │
   │  │  • Neutrino reconstruction (analytical solvers)      │  │
   │  │  • Optimizer integration (LibTorch)                  │  │
   │  │  • Metrics and ROC curves                            │  │
   │  │  • Plotting utilities                                │  │
   │  └──────────────────────────────────────────────────────┘  │
   └───────────────────┬─────────────────────────────────────────┘
                       │
   ┌───────────────────┴─────────────────────────────────────────┐
   │                External Libraries                            │
   │  • ROOT (I/O, histograms)                                   │
   │  • LibTorch (ML operations)                                 │
   │  • HDF5 (data storage)                                      │
   │  • RapidJSON (metadata)                                     │
   └─────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────┐
   │                  PyC Package (Parallel)                      │
   │  (pyc/pyc/ - High-performance C++/CUDA algorithms)         │
   │                                                              │
   │  • Physics calculations (ΔR, invariant mass)                │
   │  • Graph algorithms (kNN, PageRank, pooling)                │
   │  • Tensor operations (scatter, aggregate)                   │
   │  • Coordinate transformations                                │
   │  • CUDA-accelerated neutrino reconstruction                 │
   │  • Can be used independently of main framework              │
   └─────────────────────────────────────────────────────────────┘

Data Flow
---------

**1. Data Ingestion**

.. code-block:: text

   ROOT Files → IO Module → Data Structures → Event Templates

**2. Event Processing**

.. code-block:: text

   Event Template → Particle Registration → Event Compilation
        ↓
   Selection Template (optional) → Event Filtering
        ↓
   Graph Template (optional) → Graph Construction

**3. Analysis Execution**

.. code-block:: text

   Analysis Engine:
     - Read events from ROOT files
     - Apply templates in sequence
     - Parallel processing per file
     - Progress reporting
     - Result aggregation

**4. Output**

.. code-block:: text

   Results → Serialization → ROOT/HDF5/Pickle
          → Metrics → Evaluation
          → Plots → Visualization

Template Pattern
----------------

All template classes follow the same design pattern:

**1. C++ Base Class** (``src/AnalysisG/modules/<type>/``)

- Virtual methods for user customization
- cproperty accessors for configuration
- Core implementation

**2. Cython Wrapper** (``src/AnalysisG/core/<type>_template.pyx``)

- Exposes C++ class to Python
- Manages C++ object lifetime
- Handles type conversions

**3. User Subclass** (User code or ``src/AnalysisG/<types>/``)

- Override virtual methods
- Implement analysis logic
- Register with Analysis

Example:

.. code-block:: cpp

   // C++ Base (modules/event/include/templates/event_template.h)
   class event_template {
   public:
       virtual void build(element_t* el);
       virtual void CompileEvent();
       cproperty<std::string, event_template> tree;
   };

.. code-block:: python

   # Cython Wrapper (core/event_template.pyx)
   cdef class EventTemplate:
       cdef event_template* ptr
       
       @property
       def Tree(self):
           return self.ptr.tree

.. code-block:: python

   # User Subclass
   from AnalysisG.core.event_template import EventTemplate
   
   class MyEvent(EventTemplate):
       def __init__(self):
           super().__init__()
           self.Tree = "nominal"

Memory Management
-----------------

**C++ Objects**

- Created with ``new`` in Cython ``__cinit__``
- Deleted with ``del`` in Cython ``__dealloc__``
- Smart pointers used where appropriate
- RAII patterns for resource management

**Python Objects**

- Reference counted by Python
- Cython ensures C++ cleanup on Python object destruction
- No manual memory management needed from user perspective

**Multi-threading**

- Thread-safe template instantiation
- Per-thread ROOT file readers
- Lock-free progress reporting
- Parallel event processing

Performance Optimization
------------------------

**Zero-Copy Operations**

- Direct ROOT TTree access where possible
- Shared C++ pointers across Cython boundary
- Minimal data copying between layers

**Parallel Processing**

- Multi-threaded file reading
- Concurrent event processing
- CPU-bound operations in C++
- GPU operations in CUDA (PyC package)

**Memory Efficiency**

- Lazy evaluation of coordinate transformations
- On-demand particle instantiation
- Efficient data structures (maps, vectors)
- Memory pooling for frequently allocated objects

Extension Points
----------------

Users can extend AnalysisG at multiple levels:

**1. Template Classes** (Most Common)

- Subclass EventTemplate, ParticleTemplate, etc.
- Override virtual methods
- Add custom properties and methods

**2. C++ Modules** (Advanced)

- Add new modules in ``src/AnalysisG/modules/``
- Implement in C++ for performance
- Create Cython wrappers
- Register with Analysis engine

**3. PyC Algorithms** (Specialists)

- Add high-performance algorithms in ``pyc/pyc/``
- Implement in C++ or CUDA
- Use LibTorch for tensor operations
- Can be used standalone

Integration with ML Frameworks
-------------------------------

**PyTorch**

- Graph templates produce PyTorch Geometric Data objects
- Model templates wrap PyTorch nn.Module
- LibTorch used in C++ backend
- Seamless gradient flow

**Training Pipeline**

.. code-block:: text

   ROOT → Events → Graphs → PyTorch Data → Model → Loss → Optimizer
     ↑_____________________________↓
            Analysis Framework

**Inference Pipeline**

.. code-block:: text

   ROOT → Events → Graphs → Model → Predictions → Analysis

Directory Structure
-------------------

.. code-block:: text

   AnalysisG/
   ├── src/AnalysisG/          # Main package
   │   ├── core/               # Cython wrappers (*.pyx, *.pxd)
   │   ├── modules/            # C++ backend (*.cxx, *.h)
   │   ├── events/             # Example event implementations
   │   ├── graphs/             # Example graph implementations
   │   ├── selections/         # Example selection implementations
   │   ├── models/             # Example model implementations
   │   └── metrics/            # Example metric implementations
   ├── pyc/                    # PyC standalone package
   │   └── pyc/                # C++/CUDA algorithms
   │       ├── physics/        # Physics calculations
   │       ├── graph/          # Graph algorithms
   │       ├── operators/      # Tensor operations
   │       ├── transform/      # Coordinate transforms
   │       └── nusol/          # Neutrino reconstruction
   ├── docs/                   # Documentation
   ├── test/                   # Unit tests
   ├── studies/                # Analysis examples
   └── scripts/                # Utility scripts

Build System
------------

**CMake Configuration**

- Finds ROOT, PyTorch, CUDA, HDF5
- Generates Cython extensions
- Compiles C++ libraries
- Links everything together

**Compilation Steps**

1. CMake configures build
2. C++ modules compiled
3. Cython generates C++ from .pyx
4. Cython extensions compiled
5. Python package assembled

See Also
--------

* :doc:`installation` - Build and installation guide
* :doc:`api/index` - Complete API reference
* :doc:`examples/index` - Usage examples
