Combinatorial Neutrino Reconstruction (``selections.neutrino.combinatorial``)
=============================================================================

Import with::

    from AnalysisG.selections.neutrino.combinatorial import NuNuCombinatorial

NuNuCombinatorial
-----------------

``NuNuCombinatorial`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that performs combinatorial di-neutrino reconstruction using the NuSol
ellipse algorithm.

**Configuration**:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``num_device``
     - ``int``
     - Number of CUDA devices to use.  Default ``0`` (CPU).
   * - ``masstop``
     - ``float``
     - Top-quark mass hypothesis [GeV].
   * - ``massw``
     - ``float``
     - W-boson mass hypothesis [GeV].
   * - ``lx``
     - ``int``
     - Number of lepton–b-quark pairings to evaluate.

**Output attributes** (all accessed via the ``lx`` pairing index):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``pmu``
     - Per-pairing 4-momentum arrays (pT, η, φ, E) of input particles.
   * - ``pdgid``
     - Per-pairing PDG-ID arrays.
   * - ``matched_bq``
     - Per-pairing matched b-quark index arrays.
   * - ``matched_lp``
     - Per-pairing matched lepton index arrays.
   * - ``ellipse``
     - Per-pairing NuSol ellipse-distance values.
   * - ``chi2_nu1`` / ``chi2_nu2``
     - χ² values for the two neutrino solutions per pairing.
   * - ``pmu_nu1`` / ``pmu_nu2``
     - 4-momenta of the two neutrino solutions per pairing.

**Helper particle classes**:

* ``Neutrino`` — a :class:`~AnalysisG.core.particle_template.ParticleTemplate`
  subclass with additional fields ``ellipse``, ``chi2``, ``matched_bquark``,
  ``matched_lepton``.
* ``Particle`` — plain
  :class:`~AnalysisG.core.particle_template.ParticleTemplate`.
