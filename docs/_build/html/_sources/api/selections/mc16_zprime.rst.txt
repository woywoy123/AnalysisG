MC16 Z′ Selection (``selections.mc16.zprime``)
==============================================

Import with::

    from AnalysisG.selections.mc16.zprime import ZPrime

ZPrime
------

``ZPrime`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that collects Z′ resonance decay-product distributions for MC16 samples.

**Output attributes** (all ``list[float]``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``zprime_truth_tops``
     - pT values of truth top quarks from the Z′ decay.
   * - ``zprime_children``
     - pT values of direct Z′ decay products.
   * - ``zprime_truthjets``
     - pT values of truth jets associated with the Z′.
   * - ``zprime_jets``
     - pT values of reconstructed jets associated with the Z′.
