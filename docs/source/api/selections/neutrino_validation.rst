Neutrino Reconstruction Validation (``selections.neutrino.validation``)
========================================================================

Import with::

    from AnalysisG.selections.neutrino.validation import Validation

Validation
----------

``Validation`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that validates the NuSol static and dynamic neutrino-reconstruction
algorithms against truth-level neutrinos.

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

**Output attributes**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``Events``
     - ``list`` — reconstructed event objects (truth + dynamic + static
       neutrino solutions) populated in ``Postprocessing``.
   * - ``nu1_static`` / ``nu2_static``
     - Per-topology static neutrino 4-momenta (``dict → list[list[list[float]]]``).
   * - ``static_distances``
     - Per-topology ΔR distances for static solutions.
   * - ``nu1_dynamic`` / ``nu2_dynamic``
     - Per-topology dynamic neutrino 4-momenta.
   * - ``dynamic_distances``
     - Per-topology ΔR distances for dynamic solutions.
   * - ``pmu``
     - Per-topology input particle 4-momenta.
   * - ``pdgid``
     - Per-topology input particle PDG IDs.
   * - ``met``
     - ``list[float]`` — per-event missing-ET values.
   * - ``phi``
     - ``list[float]`` — per-event MET azimuthal angles.

**Supported pairing topologies** (used as keys in the ``dict`` outputs):

``top_children``, ``truthjet``, ``jetchildren``, ``jetleptons``.
