MC16 Decay Modes (``selections.mc16.decaymodes``)
==================================================

Import with::

    from AnalysisG.selections.mc16.decaymodes import DecayModes

DecayModes
----------

``DecayModes`` is a
:class:`~AnalysisG.core.selection_template.SelectionTemplate` subclass
that classifies the decay modes of truth top quarks in MC16 samples.

**Output attributes**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``res_top_modes``
     - ``dict[str, list[float]]`` — decay-mode fractions for resonant tops.
   * - ``res_top_charges``
     - ``dict[str, list[float]]`` — charge distributions for resonant tops.
   * - ``res_top_pdgid``
     - ``dict[str, int]`` — PDG-ID counts for resonant tops.
   * - ``spec_top_modes``
     - ``dict[str, list[float]]`` — decay-mode fractions for spectator tops.
   * - ``spec_top_charges``
     - ``dict[str, list[float]]`` — charge distributions for spectator tops.
   * - ``spec_top_pdgid``
     - ``dict[str, int]`` — PDG-ID counts for spectator tops.
   * - ``signal_region``
     - ``dict[str, list[float]]`` — signal-region discriminant distributions.
   * - ``lepton_statistics``
     - ``dict[str, int]`` — lepton-count breakdown by lepton flavour.
   * - ``all_pdgid``
     - ``dict[str, int]`` — PDG-ID counts across all decay products.
   * - ``ntops``
     - ``list[int]`` — per-event top-quark multiplicity.
