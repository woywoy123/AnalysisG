Physics Kernels
===============

Relativistic kinematics implemented as batched CUDA/C++ kernels in the
``physics_`` internal namespace and exposed via ``pyc::physics``.

Computes quantities such as squared and absolute three-momentum,
transverse momentum :math:`p_T`, pseudorapidity :math:`\eta`,
azimuthal angle :math:`\phi`, invariant mass :math:`m`, and
:math:`\Delta R`.

.. doxygennamespace:: physics_
   :project: AnalysisG
   :members:
   :undoc-members:

Doxygen Source
--------------

The documentation above is derived from the following ``.dox`` annotation file(s):

``physics.dox``

.. literalinclude:: ../../../../src/AnalysisG/pyc/physics/include/physics/physics.dox
   :language: c
   :caption:

