MC20 Top Kinematic Distributions (``selections.mc20.topkinematics``)
=====================================================================

Import with::

    from AnalysisG.selections.mc20.topkinematics import TopKinematics

TopKinematics
-------------

``TopKinematics`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that collects kinematic distributions of truth top quarks for MC20
samples.

**Output attributes** (``dict[str, list[float]]``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``res_top_kinematics``
     - Kinematic distributions for resonant tops.
   * - ``spec_top_kinematics``
     - Kinematic distributions for spectator tops.
   * - ``mass_combi``
     - Invariant-mass combinations of top pairs.
   * - ``deltaR``
     - ΔR distances between top quarks and their decay products.
