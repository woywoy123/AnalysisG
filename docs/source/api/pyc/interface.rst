pyc Interface
=============

The ``pyc`` namespace is the public C++/CUDA interface for tensor-based
particle physics computations.  All functions operate on ``torch::Tensor``
objects and are compiled into shared libraries ``libtpyc.so`` (CPU) and
``libcupyc.so`` (CUDA).  At the Python level they are registered as
``torch.ops.tpyc.*`` / ``torch.ops.cupyc.*`` custom operators.

``neutrino`` — Reconstructed Neutrino
--------------------------------------

``neutrino`` is a ``particle_template`` subclass that carries the output
of the NuSol analytical reconstruction.  In addition to the standard
kinematic properties it records which b-quark and lepton seeds were used,
the minimised :math:`\chi^2` residual, and an index into the top-quark it
belongs to.

.. doxygenclass:: neutrino
   :project: AnalysisG
   :members:
   :undoc-members:

Top-Level Utilities
-------------------

.. doxygenfunction:: pyc::std_to_dict(std::map<std::string, torch::Tensor>*)
   :project: AnalysisG

.. doxygenfunction:: pyc::tensorize(std::vector<std::vector<double>>*)
   :project: AnalysisG

Coordinate Transforms — ``pyc::transform``
-------------------------------------------

All functions accept Nx1 column tensors (separate overloads) or an Nx4
stacked tensor (combined overloads).  Inputs are validated for shape
compatibility at runtime.

**Cartesian → Polar (separate column inputs)**

- ``pyc::transform::separate::Pt(px, py)`` → :math:`p_T`
- ``pyc::transform::separate::Eta(px, py, pz)`` → :math:`\eta`
- ``pyc::transform::separate::Phi(px, py)`` → :math:`\phi`
- ``pyc::transform::separate::PtEtaPhi(px, py, pz)`` → Nx3 :math:`(p_T, \eta, \phi)`
- ``pyc::transform::separate::PtEtaPhiE(px, py, pz, e)`` → Nx4 :math:`(p_T, \eta, \phi, E)`

**Cartesian → Polar (stacked Nx4 pmc input)**

- ``pyc::transform::combined::Pt(pmc)`` → :math:`p_T`
- ``pyc::transform::combined::Eta(pmc)`` → :math:`\eta`
- ``pyc::transform::combined::Phi(pmc)`` → :math:`\phi`
- ``pyc::transform::combined::PtEtaPhi(pmc)`` → Nx3
- ``pyc::transform::combined::PtEtaPhiE(pmc)`` → Nx4

**Polar → Cartesian (separate column inputs)**

- ``pyc::transform::separate::Px(pt, phi)`` → :math:`p_x`
- ``pyc::transform::separate::Py(pt, phi)`` → :math:`p_y`
- ``pyc::transform::separate::Pz(pt, eta)`` → :math:`p_z`
- ``pyc::transform::separate::PxPyPz(pt, eta, phi)`` → Nx3
- ``pyc::transform::separate::PxPyPzE(pt, eta, phi, e)`` → Nx4

**Polar → Cartesian (stacked Nx4 pmu input)**

- ``pyc::transform::combined::Px(pmu)`` → :math:`p_x`
- ``pyc::transform::combined::PxPyPz(pmu)`` → Nx3
- ``pyc::transform::combined::PxPyPzE(pmu)`` → Nx4

See :doc:`transform` for the complete API reference.

Physics Kernels — ``pyc::physics``
-----------------------------------

The ``physics`` sub-namespace provides two coordinate variants
(``cartesian`` and ``polar``), each split into ``separate`` (per-column
tensor arguments) and ``combined`` (single stacked tensor) overloads.

**Cartesian — separate inputs** (``pyc::physics::cartesian::separate``)

- ``P2(px,py,pz)`` — squared 3-momentum :math:`|\vec{p}|^2`
- ``P(px,py,pz)`` — 3-momentum magnitude
- ``Beta2/Beta(px,py,pz,e)`` — squared/absolute velocity :math:`\beta^2`, :math:`\beta`
- ``M2/M(px,py,pz,e)`` — squared/absolute invariant mass
- ``Mt2/Mt(pz,e)`` — squared/absolute transverse mass
- ``Theta(px,py,pz)`` — polar angle :math:`\theta`
- ``DeltaR(px1,px2,py1,py2,pz1,pz2)`` — angular separation

**Cartesian — combined** (``pyc::physics::cartesian::combined``)

Same functions but accept a stacked Nx4 ``pmc`` tensor.

**Polar — separate** (``pyc::physics::polar::separate``)

- ``P2/P(pt,eta,phi)``
- ``Beta2/Beta(pt,eta,phi,e)``
- ``M2/M(pt,eta,phi,e)``
- ``Mt2/Mt(pt,eta,e)``  — note: only ``pt``, ``eta``, ``e`` needed
- ``Theta(pt,eta,phi)``
- ``DeltaR(eta1,eta2,phi1,phi2)``

**Polar — combined** (``pyc::physics::polar::combined``)

Same functions but accept a stacked Nx4 ``pmu`` tensor.

See :doc:`physics` for the complete API reference.

Matrix Operators — ``pyc::operators``
--------------------------------------

Batched linear-algebra utilities for :math:`N\times3\times3` or
:math:`N\times4\times4` tensors.

- ``Dot(v1, v2)`` — batch dot product
- ``CosTheta(v1, v2)`` / ``SinTheta(v1, v2)`` — angle between vectors
- ``Rx/Ry/Rz(angle)`` — rotation matrix construction
- ``RT(pmc_b, pmc_mu)`` — combined rotation
- ``CoFactors(matrix)`` — cofactor matrix
- ``Determinant(matrix)`` — determinant
- ``Inverse(matrix)`` → ``(inv, valid)`` — Moore–Penrose pseudo-inverse
- ``Eigenvalue(matrix)`` → ``(vals, vecs)``
- ``Cross(mat1, mat2)`` — cross product

See :doc:`operators` for the complete API reference.

Graph Aggregation — ``pyc::graph``
------------------------------------

Aggregation utilities used during GNN message-passing.

- ``edge_aggregation(edge_index, prediction, node_feature)`` — aggregate edge scores to nodes
- ``node_aggregation(edge_index, prediction, node_feature)`` — aggregate node features along edges
- ``unique_aggregation(cluster_map, features)`` — aggregate by cluster label
- ``PageRank(edge_index, edge_scores, …)`` — stable personalised PageRank
- ``PageRankReconstruction(…, pmc)`` — PageRank + 4-momentum assignment

``pyc::graph::polar`` overloads accept ``(pt, eta, phi, e)`` or a stacked
``pmu`` tensor instead of cartesian ``pmc``.

See :doc:`graph` for the complete API reference.

Neutrino Reconstruction — ``pyc::nusol``
-----------------------------------------

Batch-capable analytical neutrino reconstruction backed by the
``nusol_`` CUDA/CPU kernel namespace.

- ``BaseMatrix(pmc_b, pmc_mu, masses)`` — returns the :math:`3\times3` neutrino constraint matrix
- ``Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null)`` — single-neutrino solution
- ``NuNu(pmc_b1, pmc_b2, pmc_l1, pmc_l2, met_xy, masses, null, step, tol, timeout)`` — double-neutrino solution (uniform masses)
- ``NuNu(…, mass1, mass2, …)`` — double-neutrino solution (per-event masses)
- ``NuNu<b, l>(bquark1, bquark2, lepton1, lepton2, met, phi, mass1, mass2, dev, …)`` — particle-pointer convenience overload

See :doc:`nusol` for the complete API reference.
