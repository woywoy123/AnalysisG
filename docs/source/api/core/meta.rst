Meta / AMI (Python)
===================

Cython wrappers for ATLAS AMI dataset metadata.

ami_client
----------

``ami_client`` wraps PyAMI for ATLAS AMI queries.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``connect(endpoint: str)``
     - Connect to the given AMI endpoint (e.g. ``"atlas"``).
   * - ``get_dataset_info(dsid, amitag) → dict``
     - Fetch metadata for the dataset with the given DSID and AMI tag.

Meta
----

``Meta`` wraps the C++ ``meta_t`` struct and holds ATLAS sample metadata.

Methods
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``FetchMeta(dsid: int, amitag: str)``
     - Query AMI for metadata and populate all fields.
   * - ``GetSumOfWeights(name: str) → float``
     - Return the sum-of-weights for the given tree name.
   * - ``expected_events(lumi: float = 140.1) → float``
     - Compute expected event count as ``cross_section × lumi / 1000``.
   * - ``hash(val: str) → str``
     - Compute a tools-style hash of *val*.

Key Properties
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``dsid``
     - ``int``
     - ATLAS dataset identifier.
   * - ``amitag``
     - ``str``
     - AMI tag string (e.g. ``"p5631"``).
   * - ``generators``
     - ``str``
     - Generator string (e.g. ``"PowhegPythia8"``).
   * - ``isMC``
     - ``bool``
     - ``True`` if this is a Monte Carlo sample.
   * - ``SumOfWeights``
     - ``dict``
     - Map of tree name → sum-of-weights value.
   * - ``MetaCachePath``
     - ``str``
     - Directory for caching AMI metadata JSON files.

MetaLookup
----------

``MetaLookup`` provides a lookup table from dataset label to ``Meta``
object, populated during ``Analysis.Start()`` when ``FetchMeta=True``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``__call__(inpt) → Meta``
     - Return the ``Meta`` object corresponding to the given label or hash.

Data
----

``Data`` wraps a ``MetaLookup`` and exposes cross-section and luminosity
helpers.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Property
     - Description
   * - ``DatasetName``
     - Human-readable dataset name string.
   * - ``CrossSection``
     - Dataset cross-section in pb.
   * - ``ExpectedEvents``
     - ``CrossSection × luminosity`` (integrated luminosity in pb⁻¹).
   * - ``SumOfWeights``
     - Returns ``1`` (placeholder).
   * - ``GenerateData``
     - Returns a new ``Data`` object for this ``MetaLookup``.
