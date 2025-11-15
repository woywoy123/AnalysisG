Python-C++ Interface
====================

The PyC module provides Python bindings for AnalysisG's C++ components
using Cython.

For complete API reference, see the Doxygen-generated HTML documentation in ``doxygen-docs/html/``.

Interface Components
--------------------

C Utilities
~~~~~~~~~~~

Low-level C utility bindings.

**Location**: ``src/AnalysisG/pyc/cutils/``

Interface Layer
~~~~~~~~~~~~~~~

Main Python-C++ interface layer.

**Location**: ``src/AnalysisG/pyc/interface/``

Features:
* Core class wrappers
* Method bindings
* Property accessors
* Python-friendly APIs
* Automatic memory management
* Exception handling

Operators
~~~~~~~~~

Operator implementations for Python.

**Location**: ``src/AnalysisG/pyc/operators/``

Implements:
* Mathematical operators
* Comparison operators
* Container operators
* Special methods

Physics Module
~~~~~~~~~~~~~~

Physics calculation bindings.

**Location**: ``src/AnalysisG/pyc/physics/``

Common functions:
* Transverse momentum (pT)
* Pseudorapidity (η)
* Azimuthal angle (φ)
* ΔR distance
* Invariant mass

Transform
~~~~~~~~~

Data transformation utilities.

**Location**: ``src/AnalysisG/pyc/transform/``

Features:
* Coordinate transformations
* Reference frame changes
* Normalization operations
* Feature scaling

Graph Interface
~~~~~~~~~~~~~~~

Graph operations for Python.

**Location**: ``src/AnalysisG/pyc/graph/``

Features:
* Graph construction from Python
* Node/edge manipulation
* Graph property access
* PyTorch Geometric format conversion

Neutrino Solver Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~

Python interface to neutrino reconstruction.

**Location**: ``src/AnalysisG/pyc/nusol/``

Submodules:
* ``tensor/`` - Tensor operations
* ``cuda/`` - CUDA acceleration interface

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from AnalysisG import Event, Particle, Graph
   
   # Create event
   event = Event()
   event.load_from_file("data.root")
   
   # Access particles
   for particle in event.particles:
       print(f"pT: {particle.pt}, eta: {particle.eta}")
   
   # Build graph
   graph = Graph()
   graph.build(event)

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG import NeutrinoSolver, PhysicsUtils
   
   # Neutrino reconstruction
   solver = NeutrinoSolver()
   solutions = solver.solve(leptons, jets, met)
   
   # Physics calculations
   dr = PhysicsUtils.delta_r(particle1, particle2)
   mass = PhysicsUtils.invariant_mass([p1, p2, p3])

Design Philosophy
-----------------

The PyC interface follows:

* **Pythonic**: Natural Python syntax
* **Performance**: Minimal overhead
* **Safety**: Type checking and error handling
* **Memory**: Automatic management
* **Compatibility**: Works with NumPy, PyTorch

Cython Implementation
---------------------

Uses Cython for binding:

* ``.pxd`` files: C++ declarations
* ``.pyx`` files: Implementation
* Automatic class wrapping
* Direct memory access

Benefits:

* Near-zero overhead
* Native Python integration
* Automatic reference counting
* Exception translation

Adding New Bindings
-------------------

To add bindings for a C++ class:

1. Declare in ``.pxd`` file:

.. code-block:: cython

   cdef extern from "myclass.h":
       cdef cppclass MyClass:
           void method()
           int property

2. Wrap in ``.pyx`` file:

.. code-block:: cython

   cdef class PyMyClass:
       cdef MyClass* c_instance
       
       def method(self):
           self.c_instance.method()
       
       @property
       def property(self):
           return self.c_instance.property
