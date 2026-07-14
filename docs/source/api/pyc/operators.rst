Tensor Operators
================

Batched linear-algebra and angular operators implemented as CUDA/C++
kernels and exposed via ``pyc::operators``.

``pyc::operators`` Functions
-----------------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``Dot(v1, v2)``
     - Batch inner (dot) product of two vector tensors
   * - ``CosTheta(v1, v2)``
     - Cosine of the opening angle between ``v1`` and ``v2``
   * - ``SinTheta(v1, v2)``
     - Sine of the opening angle
   * - ``Rx(angle)``
     - Rotation matrix around x-axis for each element in ``angle``
   * - ``Ry(angle)``
     - Rotation matrix around y-axis
   * - ``Rz(angle)``
     - Rotation matrix around z-axis
   * - ``RT(pmc_b, pmc_mu)``
     - Combined rotation aligning b-quark to z-axis in the lepton rest frame
   * - ``CoFactors(matrix)``
     - Cofactor matrix of each :math:`3\times3` element in the batch
   * - ``Determinant(matrix)``
     - Determinant of each matrix in the batch
   * - ``Inverse(matrix)``
     - Returns ``(inv, valid)`` — Moore–Penrose pseudo-inverse and validity mask
   * - ``Eigenvalue(matrix)``
     - Returns ``(eigenvalues, eigenvectors)`` for symmetric matrices
   * - ``Cross(mat1, mat2)``
     - Batch 3-vector cross product

.. doxygennamespace:: pyc::operators
   :project: AnalysisG
   :members:
   :undoc-members:

Internal Kernel Namespace
--------------------------

.. doxygennamespace:: operators_
   :project: AnalysisG
   :members:
   :undoc-members:
