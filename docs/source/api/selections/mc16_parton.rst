MC16 Parton Energy Fractions (``selections.mc16.parton``)
==========================================================

Import with::

    from AnalysisG.selections.mc16.parton import Parton

Parton
------

``Parton`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that measures the fractional parton-energy contributions to truth jets
and reconstructed jets per top-quark multiplicity class.

**Output attributes** (all ``dict[str, list[float]]``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``ntops_tjets_pt``
     - pT distribution of truth jets per n-top class.
   * - ``ntops_tjets_e``
     - Energy distribution of truth jets per n-top class.
   * - ``ntops_jets_pt``
     - pT distribution of reco jets per n-top class.
   * - ``ntops_jets_e``
     - Energy distribution of reco jets per n-top class.
   * - ``nparton_tjet_e``
     - Total parton energy within each truth jet per n-parton count.
   * - ``nparton_jet_e``
     - Total parton energy within each reco jet per n-parton count.
   * - ``frac_parton_tjet_e``
     - Fraction of truth-jet energy carried by its partons.
   * - ``frac_parton_jet_e``
     - Fraction of reco-jet energy carried by its partons.
   * - ``frac_ntop_tjet_contribution``
     - Fraction of truth jets matched to exactly n top quarks.
   * - ``frac_ntop_jet_contribution``
     - Fraction of reco jets matched to exactly n top quarks.
   * - ``frac_mass_top``
     - Fractional invariant-mass residuals for top-quark candidates.
