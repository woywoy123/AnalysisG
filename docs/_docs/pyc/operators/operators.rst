operators Module
===============

The ``operators`` module provides mathematical operators for vectors and tensors commonly used in physics calculations.

Vector Operations
--------------

.. py:function:: add(vec1, vec2)

   Add two vectors component-wise.
   
   :param torch.Tensor vec1: first vector tensor
   :param torch.Tensor vec2: second vector tensor
   :return: component-wise sum of vectors
   :rtype: torch.Tensor

.. py:function:: subtract(vec1, vec2)

   Subtract second vector from first vector component-wise.
   
   :param torch.Tensor vec1: first vector tensor
   :param torch.Tensor vec2: second vector tensor
   :return: component-wise difference of vectors
   :rtype: torch.Tensor

.. py:function:: scale(vec, scalar)

   Scale a vector by a scalar value.
   
   :param torch.Tensor vec: vector tensor
   :param torch.Tensor scalar: scalar tensor
   :return: scaled vector
   :rtype: torch.Tensor

.. py:function:: magnitude(vec)

   Calculate the magnitude of a vector.
   
   :param torch.Tensor vec: vector tensor
   :return: magnitude tensor
   :rtype: torch.Tensor

.. py:function:: normalize(vec)

   Normalize a vector to unit length.
   
   :param torch.Tensor vec: vector tensor
   :return: normalized vector tensor
   :rtype: torch.Tensor

Dot and Cross Products
-------------------

.. py:function:: dot(vec1, vec2)

   Calculate the dot product between two vectors.
   
   :param torch.Tensor vec1: first vector tensor
   :param torch.Tensor vec2: second vector tensor
   :return: dot product tensor
   :rtype: torch.Tensor

.. py:function:: cross(vec1, vec2)

   Calculate the cross product between two 3D vectors.
   
   :param torch.Tensor vec1: first vector tensor
   :param torch.Tensor vec2: second vector tensor
   :return: cross product tensor
   :rtype: torch.Tensor

Angular Operations
---------------

.. py:function:: angle_between(vec1, vec2)

   Calculate the angle between two vectors.
   
   :param torch.Tensor vec1: first vector tensor
   :param torch.Tensor vec2: second vector tensor
   :return: angle between vectors in radians
   :rtype: torch.Tensor

.. py:function:: cos_angle(vec1, vec2)

   Calculate the cosine of the angle between two vectors.
   
   :param torch.Tensor vec1: first vector tensor
   :param torch.Tensor vec2: second vector tensor
   :return: cosine of angle between vectors
   :rtype: torch.Tensor

Four-Vector Operations
-------------------

.. py:function:: lorentz_boost(four_vec, boost_vec)

   Apply a Lorentz boost to a four-vector.
   
   :param torch.Tensor four_vec: four-vector tensor (E, px, py, pz)
   :param torch.Tensor boost_vec: boost vector tensor (bx, by, bz)
   :return: boosted four-vector
   :rtype: torch.Tensor

.. py:function:: four_vector_add(four_vec1, four_vec2)

   Add two four-vectors.
   
   :param torch.Tensor four_vec1: first four-vector tensor (E1, px1, py1, pz1)
   :param torch.Tensor four_vec2: second four-vector tensor (E2, px2, py2, pz2)
   :return: sum four-vector (E1+E2, px1+px2, py1+py2, pz1+pz2)
   :rtype: torch.Tensor

Examples
-------

Basic usage examples:

.. code-block:: python

   import torch
   from AnalysisG.pyc import operators
   
   # Create some vector tensors
   vec1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
   vec2 = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
   
   # Vector addition
   result = operators.add(vec1, vec2)
   print(f"Vector addition: {result}")
   
   # Vector subtraction
   result = operators.subtract(vec1, vec2)
   print(f"Vector subtraction: {result}")
   
   # Vector scaling
   scalar = torch.tensor([2.0, 3.0])
   result = operators.scale(vec1, scalar.unsqueeze(-1))
   print(f"Vector scaling: {result}")
   
   # Vector magnitude
   result = operators.magnitude(vec1)
   print(f"Vector magnitude: {result}")
   
   # Vector dot product
   result = operators.dot(vec1, vec2)
   print(f"Dot product: {result}")
   
   # Vector cross product
   result = operators.cross(vec1, vec2)
   print(f"Cross product: {result}")
   
   # Angle between vectors
   result = operators.angle_between(vec1, vec2)
   print(f"Angle between vectors (radians): {result}")
   
   # Lorentz boost example
   four_vec = torch.tensor([100.0, 10.0, 20.0, 30.0])  # (E, px, py, pz)
   boost_vec = torch.tensor([0.1, 0.2, 0.3])  # (βx, βy, βz)
   
   boosted = operators.lorentz_boost(four_vec, boost_vec)
   print(f"Boosted four-vector: {boosted}")