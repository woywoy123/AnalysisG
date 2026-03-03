Coordinate Transforms
=====================

Polar :math:`\leftrightarrow` Cartesian coordinate system transformations.
All functions operate on ``torch::Tensor`` objects and are accessible at
the Python level as ``torch.ops.tpyc.*`` / ``torch.ops.cupyc.*``.

The namespace is split into:

- ``pyc::transform::separate`` — accepts one column tensor per coordinate
- ``pyc::transform::combined`` — accepts a single stacked Nx4 tensor

The formulas below are taken directly from ``transform/transform.cxx``
(CPU) and ``transform/transform.cu`` (CUDA):

.. math::

   p_x = p_T \cos\phi, \quad
   p_y = p_T \sin\phi, \quad
   p_z = p_T \sinh\eta

   p_T = \sqrt{p_x^2 + p_y^2}, \quad
   \phi = \mathrm{atan2}(p_y, p_x), \quad
   \eta = \mathrm{asinh}\!\left(\frac{p_z}{p_T}\right)

Separate Column Inputs
-----------------------

.. doxygennamespace:: pyc::transform::separate
   :project: AnalysisG
   :members:
   :undoc-members:

Combined Stacked Input
-----------------------

.. doxygennamespace:: pyc::transform::combined
   :project: AnalysisG
   :members:
   :undoc-members:

Internal Kernel Namespace
--------------------------

The ``transform_`` internal namespace provides the raw implementations
before wrapping in the public ``pyc::transform`` API.

.. doxygennamespace:: transform_
   :project: AnalysisG
   :members:
   :undoc-members:
