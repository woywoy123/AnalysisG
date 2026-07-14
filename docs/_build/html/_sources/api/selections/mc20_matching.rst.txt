MC20 Truth Matching (``selections.mc20.matching``)
===================================================

Import with::

    from AnalysisG.selections.mc20.matching import TopMatching

TopMatching (MC20)
------------------

``TopMatching`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that performs truth-to-reco matching for MC20 samples using an energy
constraint.

**Configuration**:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``energy_constraint``
     - ``float``
     - Energy threshold used to accept or reject candidate matches.
       Default ``0.0``.
