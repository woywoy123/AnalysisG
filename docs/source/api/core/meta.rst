Meta / AMI (Python)
===================

Cython wrappers for ATLAS AMI dataset metadata.  Import from
``AnalysisG.core.meta``.

ami_client
----------

``ami_client`` wraps PyAMI authentication and dataset queries.  It is
used internally by ``Meta.FetchMeta`` and is not normally instantiated
by user code.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method (internal)
     - Description
   * - ``loadcache(Meta) → bool``
     - Load cached PyAMI query results from the HDF5 file at
       ``Meta.MetaCachePath``.  Returns ``True`` if all three caches
       (dsids, datasets, infos) were populated from disk.
   * - ``savecache(Meta)``
     - Pickle-serialise the current dsid/dataset/info query results and
       write them into the HDF5 cache file.
   * - ``dressmeta(Meta, dset_name: str)``
     - Populate ``Meta`` fields (``logicalDatasetName``, ``nFiles``,
       ``totalEvents``, ``totalSize``, ``dataType``, ``prodsysStatus``,
       ``ecmEnergy``, ``PDF``, ``genFiltEff``, ``physicsShort``,
       ``generatorName``, ``geometryVersion``, ``conditionsTag``,
       ``generatorTune``, ``amiStatus``, ``beamType``, ``productionStep``,
       ``projectName``, ``statsAlgorithm``, ``beam_energy``,
       ``genFilterNames``, ``principalPhysicsGroup``, ``kfactor``,
       ``weights``, ``keywords``, ``keyword``, ``fileGUID``, ``fileSize``,
       ``events``) from the AMI ``get_dataset_info`` / ``list_files``
       response for the named dataset.
   * - ``list_datasets(Meta)``
     - Query AMI for all datasets matching ``Meta.dsid`` and
       ``Meta.amitag``; call ``dressmeta`` for the first dataset whose
       tag matches; write back to the HDF5 cache on a cache miss.

Meta
----

``Meta`` wraps the C++ ``meta_t`` struct and exposes all ATLAS sample
metadata fields as Python properties.

.. code-block:: python

   from AnalysisG.core.meta import Meta

   m = Meta()
   m.MetaCachePath = "/data/metacache.hdf5"
   m.FetchMeta(410470, "e7101_s3681_r13144_p5855")

   print(m.dsid, m.ecmEnergy, m.generators)
   print("cross-section [pb]:", m.crossSection)
   print("expected events at 140 fb-1:", m.expected_events(140.0))

Methods
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``FetchMeta(dsid: int, amitag: str)``
     - Set ``self.dsid`` and ``self.amitag``, then call
       ``ami_client.list_datasets`` to query PyAMI and populate all
       metadata fields.
   * - ``GetSumOfWeights(name: str) → float``
     - Return ``misc[name].processed_events_weighted``, or ``1`` if
       zero / absent.
   * - ``expected_events(lumi: float = 140.1) → float``
     - Compute expected event count as ``crossSection × lumi``.
       Returns ``0`` when ``crossSection < 0``.
   * - ``hash(val: str) → str``
     - Return the 18-character hex hash of the last ``"/"``-separated
       component of *val* (same as ``tools.hash(basename(val))``).

Boolean Properties
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Description
   * - ``isMC``
     - ``True`` if the dataset is Monte Carlo simulation (default
       ``True``).
   * - ``found``
     - ``True`` after a successful AMI query (set inside
       ``dressmeta``).

Scalar Numeric Properties
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Type
     - Description
   * - ``dsid``
     - ``int``
     - ATLAS Dataset Identifier (e.g. ``410470``).
   * - ``datasetNumber``
     - ``int``
     - AMI internal dataset number; may differ from ``dsid`` for
       derived datasets.
   * - ``nFiles``
     - ``int``
     - Number of files in the dataset (from AMI).
   * - ``totalEvents``
     - ``int``
     - Total number of events as reported by AMI.
   * - ``eventNumber``
     - ``float``
     - ROOT-level event-number counter; ``-1`` until scanned.
   * - ``event_index``
     - ``int``
     - Framework event position index; ``-1`` until set.
   * - ``totalSize``
     - ``float``
     - Total dataset size in bytes (from AMI).
   * - ``kfactor``
     - ``float``
     - Higher-order QCD/EW K-factor from PMG (default ``1``).
   * - ``ecmEnergy``
     - ``float``
     - Centre-of-mass energy in GeV (e.g. ``13000.0``).
   * - ``genFiltEff``
     - ``float``
     - Generator filter efficiency (0–1).
   * - ``completion``
     - ``float``
     - Dataset completion fraction (0–1) from AMI.
   * - ``beam_energy``
     - ``float``
     - Individual beam energy in GeV (half of ``ecmEnergy``).
   * - ``crossSection``
     - ``float``
     - Inclusive cross-section in **pb** (``meta_t.crossSection_mean
       × 10⁶``).
   * - ``crossSection_mean``
     - ``float``
     - Raw cross-section in nb as stored in ``meta_t``.
   * - ``campaign_luminosity``
     - ``float``
     - Integrated luminosity of the ATLAS campaign in fb⁻¹.

String Properties
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Property
     - Description
   * - ``amitag``
     - AMI production tag string (e.g. ``"e7101_s3681_r13144_p5855"``).
       Read from ``inputConfig.amiTag`` or parsed from the file path.
   * - ``derivationFormat``
     - ATLAS DAOD derivation format (e.g. ``"TOPQ1"``).
   * - ``generators``
     - Space-separated generator names (e.g. ``"PP8"`` for Powheg+Pythia8).
       The setter replaces ``"+"`` with ``" "``.
   * - ``identifier``
     - AMI dataset identifier string.
   * - ``DatasetName``
     - Human-readable logical dataset name.
   * - ``logicalDatasetName``
     - Full AMI logical dataset name including all processing tags
       (e.g. ``"mc21_13p6TeV.410470.PhPy8EG_ttbar.deriv.DAOD_TOPQ1.…"``).
   * - ``prodsysStatus``
     - Production system status (e.g. ``"done"``).
   * - ``dataType``
     - Data format type (e.g. ``"DAOD"``).
   * - ``version``
     - Dataset version string (e.g. ``"001"``).
   * - ``PDF``
     - Parton distribution function set name (e.g. ``"PDF4LHC21"``).
   * - ``AtlasRelease``
     - ATLAS software release tag (e.g. ``"AthDerivation-22.2.96"``).
   * - ``principalPhysicsGroup``
     - ATLAS physics group responsible for the sample (e.g. ``"Top"``).
   * - ``physicsShort``
     - Short physics-process description label
       (e.g. ``"ttbar_Pythia8EvtGen_A14_NNPDF23_AFII"``).
   * - ``generatorName``
     - Full generator name string (e.g. ``"Powheg+Pythia8"``).
   * - ``geometryVersion``
     - ATLAS detector geometry tag
       (e.g. ``"ATLAS-R3S-2021-03-02-00"``).
   * - ``conditionsTag``
     - ATLAS conditions database tag
       (e.g. ``"OFLCOND-MC21-SDR-RUN3-09"``).
   * - ``generatorTune``
     - MC underlying-event tune name
       (e.g. ``"A14 NNPDF2.3 AFII"``).
   * - ``amiStatus``
     - AMI dataset lifecycle status (e.g. ``"valid"``).
   * - ``beamType``
     - Beam collision type (e.g. ``"collisions"``).
   * - ``productionStep``
     - AMI production step label (e.g. ``"merge"``).
   * - ``projectName``
     - ATLAS MC production project (e.g. ``"mc21_13p6TeV"``).
   * - ``statsAlgorithm``
     - Statistical algorithm for event counting
       (e.g. ``"EventCount"``).
   * - ``genFilterNames``
     - Comma-separated generator filter names applied to the sample.
   * - ``file_type``
     - File format type string (e.g. ``"DAOD_TOPQ1"``).
   * - ``sample_name``
     - Full sample path or name; used as a fallback source for
       ``amitag`` when ``AnalysisTracking`` lacks ``inputConfig.amiTag``.
   * - ``campaign``
     - ATLAS campaign identifier extracted from the sum-of-weights
       histogram (e.g. ``"mc21a"``), with spaces stripped.
   * - ``MetaCachePath``
     - Path to the HDF5 file used to cache AMI query results.

List Properties
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Property
     - Element type
     - Description
   * - ``keywords``
     - ``str``
     - AMI keyword list (from ``"keywords"`` split on ``", "``).
   * - ``keyword``
     - ``str``
     - Short AMI keyword list (from ``"keyword"`` split on ``", "``).
   * - ``weights``
     - ``str``
     - Available event-weight names (from ``"weights"`` split on
       ``" | "``).
   * - ``fileGUID``
     - ``str``
     - GUID strings for each input file.
   * - ``events``
     - ``int``
     - Number of events per input file.
   * - ``run_number``
     - ``int``
     - Run numbers present in the dataset.
   * - ``fileSize``
     - ``float``
     - Size of each input file in bytes.

Dict / Map Properties
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Property
     - Type
     - Description
   * - ``Files``
     - ``dict[int, str]``
     - Maps file index → local file path.
   * - ``config``
     - ``dict[str, str]``
     - Arbitrary key-value configuration pairs from the
       ``configSettings`` JSON array.
   * - ``SumOfWeights``
     - ``dict[str, weights_t]``
     - Maps tree name → ``weights_t`` struct containing
       ``processed_events_weighted``, ``dsid``, ``isAFII``,
       ``total_events``, etc.

MetaLookup
----------

``MetaLookup`` provides a lookup table from dataset label/hash to ``Meta``
object.  Populated during ``Analysis.Start()`` when ``FetchMeta = True``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Attribute / Method
     - Description
   * - ``luminosity``
     - Reference integrated luminosity in fb⁻¹ used for
       ``ExpectedEvents`` calculations (default ``140.1``).
   * - ``metadata``
     - ``dict[str, Meta]`` — internal hash-keyed store populated by
       the framework.
   * - ``matched``
     - ``dict[str, Meta]`` — cache of previously resolved
       label → Meta matches.
   * - ``__call__(inpt)``
     - Resolve a filename or label to its ``Meta`` object using the
       C++ ``meta.hash`` of the basename.
   * - ``DatasetName``
     - ``str`` — ``DatasetName`` of the last matched ``Meta``.
   * - ``CrossSection``
     - ``float`` — cross-section in pb of the last matched ``Meta``.
   * - ``ExpectedEvents``
     - ``float`` — ``CrossSection × luminosity``.
   * - ``SumOfWeights``
     - Returns ``1`` (placeholder; per-event weights are carried
       in ``Data``).
   * - ``GenerateData``
     - Returns a fresh ``Data`` object wrapping this ``MetaLookup``.

Data
----

``Data`` accumulates per-sample event-data vectors and applies
cross-section normalisation when data and weights are assigned.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Property / Method
     - Description
   * - ``data`` (setter)
     - Assign a ``dict[str, list[float]]`` (filename → values).
       Each entry is appended to the internal per-file data store
       keyed by the AMI dataset name.
   * - ``data`` (getter)
     - Concatenate all accumulated data vectors into a single
       ``vector[float]``.
   * - ``weights`` (setter)
     - Assign a ``dict[str, list[float]]`` of per-file MC weights.
       Internally normalises by sum-of-weights and cross-section.
   * - ``weights`` (getter)
     - Return the cross-section–normalised weight vector:
       ``w_i × (σ × L) / ΣW`` per dataset.
   * - ``__add__(Data) → Data``
     - Merge *other*'s weights and data into this object.
   * - ``__radd__(other)``
     - Supports ``sum([d1, d2, …], 0)`` idiom.
