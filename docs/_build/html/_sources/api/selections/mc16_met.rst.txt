MC16 Missing ET / Neutrino Reconstruction (``selections.mc16.met``)
====================================================================

Import with::

    from AnalysisG.selections.mc16.met import MissingET

MissingET
---------

``MissingET`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that runs neutrino-based missing-ET reconstruction on MC16 samples.

**Configurable parameters** (set before calling ``Start()``):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``masstop``
     - ``float``
     - Top-quark mass hypothesis [GeV].  Default ``172.5``.
   * - ``massw``
     - ``float``
     - W-boson mass hypothesis [GeV].  Default ``80.4``.
   * - ``perturb``
     - ``float``
     - Perturbation step size for the neutrino scan.
   * - ``distance``
     - ``float``
     - Maximum allowed distance for neutrino–lepton matching.
   * - ``steps``
     - ``int``
     - Number of scan steps.

**Output attribute**:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``data``
     - ``list[container_t]``
     - One ``container_t`` per event.  Each struct contains truth
       neutrino momenta, reconstructed neutrino solutions, chi² values,
       angular plane-intersection vectors, and ΔR distances between
       neutrino and lepton candidates.  See ``missing_et.pxd`` for the
       full ``container_t`` field listing.
