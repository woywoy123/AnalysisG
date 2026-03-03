Neutrino Solution (NuSol) Module
================================

The NuSol sub-package provides analytical single and double neutrino
reconstruction using ellipse intersection methods.  It exposes both a
high-level C++ ``nusol`` class and a batch-capable CUDA/LibTorch interface
(``nusol_`` namespace — see :doc:`../pyc/nusol`).

Entry Point
-----------

``nusol_enum`` selects between the two reconstruction back-ends:

.. doxygenenum:: nusol_enum
   :project: AnalysisG

.. doxygenstruct:: nusol_t
   :project: AnalysisG
   :members:

.. doxygenclass:: nusol
   :project: AnalysisG
   :members:
   :undoc-members:

Ellipse Utilities
-----------------

``mtx`` is a minimal dense matrix class used throughout the analytical
ellipse-intersection solver.  All matrix operations are implemented without
external BLAS dependencies.

.. doxygenclass:: mtx
   :project: AnalysisG
   :members:
   :undoc-members:

Constrained Neutrino Solutions
-------------------------------

``conuix`` orchestrates multi-neutrino reconstruction by creating one
``conuic`` instance per lepton–b-jet pair and merging the solutions.

.. doxygenclass:: conuix
   :project: AnalysisG
   :members:
   :undoc-members:

``conuic`` implements the single-neutrino constrained solver.  It finds the
neutrino momentum that simultaneously satisfies the W-mass and top-mass
constraints using the Möbius-parametrisation approach.

.. doxygenclass:: conuic
   :project: AnalysisG
   :members:
   :undoc-members:

Conuix Internal Structs (``Conuix`` namespace)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: atomics_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: Conuix::kinematic_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: Conuix::rotation_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: Conuix::base_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: Conuix::pencil_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: Conuix::Sx_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: Conuix::Sy_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: Conuix::H_matrix_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: Conuix::Mobius_t
   :project: AnalysisG
   :members:

ODE Runge–Kutta Multi-Neutrino Solver
--------------------------------------

``odeRK`` implements a fourth-order Runge–Kutta ODE integrator used by the
multi-solution ``multisol`` back-end to evolve ellipse states until
convergence.

.. doxygenclass:: odeRK
   :project: AnalysisG
   :members:
   :undoc-members:

.. doxygenstruct:: ellipse_t
   :project: AnalysisG
   :members:

.. doxygenstruct:: recon_t
   :project: AnalysisG
   :members:
