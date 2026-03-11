MC16 Children Kinematics (``selections.mc16.childrenkinematics``)
==================================================================

Import with::

    from AnalysisG.selections.mc16.childrenkinematics import ChildrenKinematics

ChildrenKinematics
------------------

``ChildrenKinematics`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that collects kinematic distributions of top-quark decay products for
MC16 samples.

**Output attributes** (``dict[str, list[float]]``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``res_kinematics``
     - Kinematic distributions for children from resonant tops.
   * - ``spec_kinematics``
     - Kinematic distributions for children from spectator tops.
   * - ``res_pdgid_kinematics``
     - PDG-ID-split kinematics for resonant top children.
   * - ``spec_pdgid_kinematics``
     - PDG-ID-split kinematics for spectator top children.
   * - ``res_decay_mode``
     - Decay-mode distributions for resonant tops.
   * - ``spec_decay_mode``
     - Decay-mode distributions for spectator tops.
   * - ``fractional``
     - Fractional energy / pT distributions.
   * - ``dr_clustering``
     - ΔR between clustered objects.
   * - ``top_children_dr``
     - ΔR between top and its decay products.
