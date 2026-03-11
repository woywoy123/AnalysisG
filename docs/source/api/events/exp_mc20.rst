Experimental MC20 Event (``exp_mc20``)
=======================================

The ``exp_mc20`` package provides the event and particle classes for the
experimental MC20 ATLAS dataset.  Import with::

    from AnalysisG.events.exp_mc20 import ExpMC20

ExpMC20
-------

``ExpMC20`` is an :class:`~AnalysisG.core.event_template.EventTemplate`
subclass wrapping ``<exp_mc20/event.h>``.

**Particle collections**:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``Tops``
     - ``list`` — truth top quarks.
   * - ``TruthChildren``
     - ``list`` — truth-level top decay products.
   * - ``PhysicsTruth``
     - ``list`` — truth-level physics objects (jets/leptons/photons).
   * - ``Jets``
     - ``list`` — reconstructed jets.
   * - ``Leptons``
     - ``list`` — reconstructed leptons.
   * - ``PhysicsDetector``
     - ``list`` — detector-level physics objects.
   * - ``Detector``
     - ``list`` — full detector-level collection.

**Scalar fields**:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``event_number``
     - ``int``
     - Unique 64-bit event identifier.
   * - ``met_sum``
     - ``float``
     - Scalar sum of all transverse energies [GeV].
   * - ``met``
     - ``float``
     - Missing transverse energy [GeV].
   * - ``phi``
     - ``float``
     - MET azimuthal angle [rad].
   * - ``mu``
     - ``float``
     - Average pile-up interactions per bunch crossing.

Particle Classes
----------------

Top
^^^

Truth top quark with barcode tracking.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``barcode``
     - ``int``
     - MC generator particle barcode.
   * - ``status``
     - ``int``
     - MC generator status code.

Child
^^^^^

Truth-level top decay product.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``barcode``
     - ``int``
     - MC generator particle barcode.
   * - ``status``
     - ``int``
     - MC generator status code.

PhysicsDetector / PhysicsTruth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unified detector / truth physics objects (jets, leptons, photons).

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``parton_label``
     - ``int``
     - Parton-level flavour label.
   * - ``cone_label``
     - ``int``
     - Cone-based flavour label.
   * - ``is_jet``
     - ``bool``
     - Object is classified as a jet.
   * - ``is_lepton``
     - ``bool``
     - Object is classified as a lepton.
   * - ``is_photon``
     - ``bool``
     - Object is classified as a photon.

Electron
^^^^^^^^

Reconstructed electron with quality flags.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``d0``
     - ``float``
     - Transverse impact parameter [mm].
   * - ``delta_z0``
     - ``float``
     - Longitudinal impact parameter [mm].
   * - ``true_type``
     - ``int``
     - MC truth type.
   * - ``true_origin``
     - ``int``
     - MC truth origin.
   * - ``is_tight``
     - ``bool``
     - Passes the tight electron selection.

Muon
^^^^

Reconstructed muon with quality flags.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``d0``
     - ``float``
     - Transverse impact parameter [mm].
   * - ``delta_z0``
     - ``float``
     - Longitudinal impact parameter [mm].
   * - ``true_type``
     - ``int``
     - MC truth type.
   * - ``true_origin``
     - ``int``
     - MC truth origin.
   * - ``is_tight``
     - ``bool``
     - Passes the tight muon selection.

Jet
^^^

Reconstructed jet with b-tagging working points.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``btag_65`` / ``btag_70`` / ``btag_77`` / ``btag_85`` / ``btag_90``
     - ``bool``
     - b-tag flags at 65 / 70 / 77 / 85 / 90 % working points.
   * - ``flav``
     - ``int``
     - Jet flavour label.
   * - ``label``
     - ``int``
     - Jet label index.
