Neutrino Reconstruction (pyc)
==============================

Analytical single and double neutrino reconstruction kernels implemented
as batched CUDA/C++ operations.  This module bridges the high-level
``nusol`` C++ module (see :doc:`../modules/nusol`) with the PyTorch tensor
interface.

``pyc::nusol`` Functions
--------------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``BaseMatrix(pmc_b, pmc_mu, masses)``
     - Computes the :math:`3\times3` neutrino constraint matrix from b-quark, lepton, and mass hypothesis tensors.  Returns a dict with key ``"H"``.
   * - ``Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null)``
     - Single-neutrino analytical solution (ellipse intersection).  Returns a dict with ``"nu_pmc"`` (Nx4), ``"chi2"`` (Nx1), and ``"is_valid"`` (Nx1).
   * - ``NuNu(pmc_b1, pmc_b2, pmc_l1, pmc_l2, met_xy, masses, null, step, tol, timeout)``
     - Double-neutrino solution assuming uniform mass hypotheses across the batch.
   * - ``NuNu(…, mass1, mass2, …)``
     - Double-neutrino solution with per-event mass hypothesis tensors.
   * - ``NuNu<b,l>(bquark1, …, dev, null, step, tol, timeout)``
     - Particle-pointer overload — accepts ``std::vector<particle_template*>`` for use in C++ selections.

``nusol_`` Internal Namespace
-------------------------------

The ``nusol_`` namespace contains the raw CUDA/CPU kernel implementations
used by ``pyc::nusol``:

- ``BaseMatrix`` — three overloads: tensor masses, scalar masses (mT/mW), null check
- ``Hperp`` — perpendicular H matrix
- ``Intersection`` — ellipse intersection routine
- ``Nu`` — single neutrino kernel
- ``NuNu`` — double neutrino kernel (two overloads + iterative solver)
- ``combinatorial`` — combinatorial neutrino assignment with PageRank

.. doxygennamespace:: pyc::nusol
   :project: AnalysisG
   :members:
   :undoc-members:

.. doxygennamespace:: nusol_
   :project: AnalysisG
   :members:
   :undoc-members:
