Tools Module (PyC)
==================

Utility functions for PyC package.

Overview
--------

Located in ``pyc/pyc/tools/``, this module provides utilities for data handling, 
I/O, and conversions.

Key Functions
-------------

I/O Operations
~~~~~~~~~~~~~~

.. code-block:: python

   import pyc.tools as tools
   
   # Read data
   data = tools.read_file("data.bin")
   
   # Write data
   tools.write_file("output.bin", data)

Type Conversions
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert between types
   tensor = tools.list_to_tensor([1, 2, 3, 4])
   list_data = tools.tensor_to_list(tensor)

String Utilities
~~~~~~~~~~~~~~~~

.. code-block:: python

   # String operations
   encoded = tools.encode_string("text")
   decoded = tools.decode_string(encoded)

See Also
--------

* :doc:`physics` - Physics calculations
* :doc:`operators` - Tensor operations
