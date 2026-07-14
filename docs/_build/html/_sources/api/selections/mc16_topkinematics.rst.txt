MC16 Top Kinematic Distributions (``selections.mc16.topkinematics``)
=====================================================================

Import with::

    from AnalysisG.selections.mc16.topkinematics import TopKinematics

TopKinematics
-------------

``TopKinematics`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that collects kinematic distributions of truth top quarks in MC16 samples.

**Output attributes** (``dict[str, list[float]]``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``res_top_kinematics``
     - Kinematic distributions (pT, η, φ, m, E …) for resonant tops.
   * - ``spec_top_kinematics``
     - Kinematic distributions for spectator tops.
   * - ``mass_combi``
     - Invariant mass of various top-quark combinations.
   * - ``deltaR``
     - ΔR between top pairs and between a top and its decay products.
