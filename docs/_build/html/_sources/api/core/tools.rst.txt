Tools Module
============

The tools module provides utility functions and helper classes used throughout AnalysisG.

Overview
--------

The ``tools`` module (``src/AnalysisG/core/tools.pyx``) contains:

- String encoding/decoding utilities
- Type conversion functions
- Data structure helpers
- Common operations

These utilities are used internally by AnalysisG and are also available for user code.

API Reference
-------------

String Utilities
~~~~~~~~~~~~~~~~

.. function:: enc(s: str) -> bytes

   Encode Python string to C++ string.

   :param s: Python string
   :type s: str
   :return: Encoded bytes
   :rtype: bytes

   Used internally for Python-C++ string conversion.

.. function:: env(b: bytes) -> str

   Decode C++ string to Python string.

   :param b: Bytes from C++
   :type b: bytes
   :return: Python string
   :rtype: str

   Used internally for C++-Python string conversion.

Type Conversion
~~~~~~~~~~~~~~~

The tools module provides type conversion utilities for:

- ROOT types to Python types
- Numpy arrays to C++ vectors
- Python lists to C++ vectors
- And vice versa

Data Structure Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Common operations on data structures:

- Flattening nested lists
- Filtering collections
- Mapping transformations
- Aggregation operations

Physics Utilities
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import tools
   
   # Calculate delta R
   delta_r = tools.DeltaR(eta1, phi1, eta2, phi2)
   
   # Calculate invariant mass
   mass = tools.InvariantMass(pt1, eta1, phi1, E1, pt2, eta2, phi2, E2)

Vector Operations
~~~~~~~~~~~~~~~~~

Operations on 4-vectors and 3-vectors:

.. code-block:: python

   from AnalysisG.core import tools
   
   # Create 4-vector
   vec = tools.FourVector(pt, eta, phi, E)
   
   # Vector operations
   pt = vec.Pt()
   eta = vec.Eta()
   phi = vec.Phi()
   mass = vec.M()

Implementation Details
----------------------

Cython Implementation
~~~~~~~~~~~~~~~~~~~~~

The tools module is implemented in Cython for:

- Performance-critical operations
- Zero-copy data transfer
- Efficient type conversion
- C++ STL integration

Common Use Cases
----------------

String Handling
~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.tools import enc, env
   
   # Convert Python string to C++ string
   cpp_str = enc("my_tree_name")
   
   # Convert C++ string back to Python
   py_str = env(cpp_str)

Working with ROOT Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import tools
   
   # Convert ROOT vector to Python list
   python_list = tools.convert_vector(root_vector)
   
   # Convert Python list to ROOT vector
   root_vector = tools.to_vector(python_list)

See Also
--------

* :doc:`analysis` - Analysis class
* :doc:`templates` - Template classes
* :doc:`io` - I/O utilities
