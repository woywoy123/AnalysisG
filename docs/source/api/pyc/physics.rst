Physics Kernels
===============

Relativistic kinematics implemented as batched CUDA/C++ kernels.  All
functions are available through the ``pyc::physics`` C++ API
(see :doc:`interface`) and as ``torch.ops.tpyc`` / ``torch.ops.cupyc``
Python operators.

The namespace is organised into two coordinate-system variants:

- ``pyc::physics::cartesian`` — inputs as Cartesian :math:`(p_x, p_y, p_z, E)`
- ``pyc::physics::polar`` — inputs as polar :math:`(p_T, \eta, \phi, E)`

Each variant exposes ``separate`` (per-column tensor arguments) and
``combined`` (single stacked Nx4 tensor) overloads.

The formulas below are taken directly from ``physics/physics.cxx``
(CPU) and ``physics/physics.cu`` (CUDA):

.. math::

   |\vec{p}|^2 &= p_x^2 + p_y^2 + p_z^2 \\
   |\vec{p}|   &= \sqrt{|\vec{p}|^2} \\
   \beta^2     &= \frac{|\vec{p}|^2}{E^2}, \quad \beta = \frac{|\vec{p}|}{E} \\
   m^2         &= E^2 - |\vec{p}|^2, \quad m = \mathrm{sign}(m^2)\sqrt{|m^2|} \\
   m_T^2       &= E^2 - p_z^2, \quad m_T = \mathrm{sign}(m_T^2)\sqrt{|m_T^2|} \\
   \theta       &= \mathrm{atan2}\!\left(\sqrt{p_x^2+p_y^2},\, p_z\right) \\
   \Delta R    &= \sqrt{(\Delta\eta)^2 + (\Delta\phi)^2}

Cartesian -- Separate Inputs
----------------------------

.. doxygennamespace:: pyc::physics::cartesian::separate
   :project: AnalysisG
   :members:
   :undoc-members:

Cartesian -- Combined Input
---------------------------

.. doxygennamespace:: pyc::physics::cartesian::combined
   :project: AnalysisG
   :members:
   :undoc-members:

Polar -- Separate Inputs
------------------------

.. doxygennamespace:: pyc::physics::polar::separate
   :project: AnalysisG
   :members:
   :undoc-members:

Polar -- Combined Input
-----------------------

.. doxygennamespace:: pyc::physics::polar::combined
   :project: AnalysisG
   :members:
   :undoc-members:

Internal Kernel Namespace
--------------------------

The ``physics_`` internal namespace provides the raw implementations
before wrapping in the public ``pyc::physics`` API.

.. doxygennamespace:: physics_
   :project: AnalysisG
   :members:
   :undoc-members:
