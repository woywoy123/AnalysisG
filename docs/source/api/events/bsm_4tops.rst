BSM 4-Tops Event (``bsm_4tops``)
=================================

The ``bsm_4tops`` package provides the event and particle classes for the
Beyond-Standard-Model four-top-quark analysis.  Import with::

    from AnalysisG.events.bsm_4tops import BSM4Tops

.. _bsm4tops-event:

BSM4Tops
--------

.. autoclass-like table for BSM4Tops

``BSM4Tops`` is a :class:`~AnalysisG.core.event_template.EventTemplate`
subclass that wraps the ``bsm_4tops`` C++ event class (``<bsm_4tops/event.h>``).

**Particle collections** (each element is a particle object listed below):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``Tops``
     - ``list`` — truth top quarks.
   * - ``Children``
     - ``list`` — direct decay products (top children).
   * - ``TruthJets``
     - ``list`` — truth-level jets.
   * - ``Jets``
     - ``list`` — reconstructed jets.
   * - ``Electrons``
     - ``list`` — reconstructed electrons.
   * - ``Muons``
     - ``list`` — reconstructed muons.
   * - ``DetectorObjects``
     - ``list`` — merged detector-level objects.

**Scalar fields** (read/write properties):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``event_number``
     - ``int``
     - Unique event number (64-bit unsigned integer).
   * - ``reconstruct_nunu``
     - ``bool``
     - When ``True`` the neutrino-pair reconstruction is run during
       ``CompileEvent``.  Default ``False``.
   * - ``mu``
     - ``float``
     - Average interactions per bunch crossing (pile-up μ).
   * - ``met``
     - ``float``
     - Missing transverse energy magnitude [GeV].
   * - ``phi``
     - ``float``
     - Missing transverse energy azimuthal angle [rad].

.. _bsm4tops-particles:

Particle Classes
----------------

top
^^^

Truth top quark.  Inherits all :class:`~AnalysisG.core.particle_template.ParticleTemplate`
kinematics.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``from_res``
     - ``bool``
     - ``True`` when the top originates from a resonance (e.g. Z′).
   * - ``status``
     - ``int``
     - MC generator status code.
   * - ``TruthJets``
     - ``list``
     - Associated truth-jet objects.
   * - ``Jets``
     - ``list``
     - Associated reconstructed-jet objects.

top_children
^^^^^^^^^^^^

Direct decay products of a truth top quark.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``top_index``
     - ``int``
     - Index of the parent top in the event ``Tops`` collection.
   * - ``from_res``
     - ``bool``
     - ``True`` when the parent top originates from a resonance.

truthjet
^^^^^^^^

Truth-level jet (particle-flow jet before detector simulation).

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``top_quark_count``
     - ``int``
     - Number of top quarks that contributed partons to this truth jet.
   * - ``w_boson_count``
     - ``int``
     - Number of W bosons that contributed partons to this truth jet.
   * - ``from_res``
     - ``bool``
     - ``True`` when at least one contributing top originates from a resonance.
   * - ``top_index``
     - ``list[int]``
     - Indices of associated top quarks.
   * - ``Tops``
     - ``list``
     - Associated ``top`` objects.
   * - ``Parton``
     - ``list``
     - Associated ``truthjetparton`` objects.

truthjetparton
^^^^^^^^^^^^^^

Parton matched to a truth jet.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``truthjet_index``
     - ``int``
     - Index into the event ``TruthJets`` collection.
   * - ``topchild_index``
     - ``list[int]``
     - Indices of associated ``top_children`` objects.

jet
^^^

Reconstructed jet with DL1/DL1r b-tagging discriminants.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``Tops``
     - ``list``
     - Associated ``top`` objects.
   * - ``Parton``
     - ``list``
     - Associated ``jetparton`` objects.
   * - ``top_index``
     - ``list[int]``
     - Indices of matched top quarks.
   * - ``btag_DL1r_60``
     - ``bool``
     - DL1r b-tag at 60 % working point.
   * - ``btag_DL1_60``
     - ``bool``
     - DL1 b-tag at 60 % working point.
   * - ``btag_DL1r_70``
     - ``bool``
     - DL1r b-tag at 70 % working point.
   * - ``btag_DL1_70``
     - ``bool``
     - DL1 b-tag at 70 % working point.
   * - ``btag_DL1r_77``
     - ``bool``
     - DL1r b-tag at 77 % working point.
   * - ``btag_DL1_77``
     - ``bool``
     - DL1 b-tag at 77 % working point.
   * - ``btag_DL1r_85``
     - ``bool``
     - DL1r b-tag at 85 % working point.
   * - ``btag_DL1_85``
     - ``bool``
     - DL1 b-tag at 85 % working point.
   * - ``DL1_b`` / ``DL1_c`` / ``DL1_u``
     - ``float``
     - Raw DL1 b/c/light-quark scores.
   * - ``DL1r_b`` / ``DL1r_c`` / ``DL1r_u``
     - ``float``
     - Raw DL1r b/c/light-quark scores.

jetparton
^^^^^^^^^

Parton matched to a reconstructed jet.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``jet_index``
     - ``int``
     - Index into the event ``Jets`` collection.
   * - ``topchild_index``
     - ``list[int]``
     - Indices of associated ``top_children`` objects.

electron / muon
^^^^^^^^^^^^^^^

Reconstructed electron and muon classes.  Both inherit all
:class:`~AnalysisG.core.particle_template.ParticleTemplate` kinematics;
no additional fields are defined beyond the base class.
