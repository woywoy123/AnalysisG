Meta Module
===========

The ``meta`` class reads and caches ATLAS AMI/AGIS dataset metadata from ROOT
files and from a local JSON cache.  It exposes all metadata fields as typed
``cproperty`` accessors so that user code can transparently read them.

Class: ``meta``
---------------

**Header:** ``<meta/meta.h>``

**Inheritance:** ``tools``, ``notification``

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``const folds_t* get_tags(std::string hash)``
     - Returns the ``folds_t`` tag struct for a given event hash.
   * - ``void scan_data(TObject* obj)``
     - Scans a ROOT ``TObject`` (typically a ``TTree`` or ``TH1F``) for
       metadata leaves (run number, event number, sum-of-weights, …).
   * - ``void scan_sow(TObject* obj)``
     - Scans a ROOT object specifically for the sum-of-weights histogram.
   * - ``void parse_json(std::string inpt)``
     - Parses an AMI JSON string into ``meta_data``.
   * - ``std::string hash(std::string fname)``
     - Returns the 18-character hash for the given filename, used as the
       dataset identifier.

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Type
     - Description
   * - ``rpd``
     - ``rapidjson::Document*``
     - Parsed JSON document (``nullptr`` until ``parse_json`` is called).
   * - ``metacache_path``
     - ``std::string``
     - Path of the local JSON metadata cache directory.
   * - ``meta_data``
     - ``meta_t``
     - Raw metadata struct (backing store for all ``cproperty`` accessors).

Boolean Properties
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Description
   * - ``isMC``
     - ``true`` if the dataset is Monte Carlo simulation.
   * - ``found``
     - ``true`` if metadata was successfully retrieved from AMI/AGIS.

Numeric Properties (``double``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - ``eventNumber``
     - Reserved internal counter that maps to a ROOT-level event-number
       field.  Populated by ``scan_data`` from the ``MetaData`` tree;
       default ``-1``.
   * - ``event_index``
     - Free-parameter index used by the framework to track the current
       event position within a dataset; default ``-1``.
   * - ``totalSize``
     - Total dataset size in bytes as reported by AMI
       (``"totalSize"`` field).
   * - ``kfactor``
     - K-factor (higher-order QCD/EW correction scale factor) for the
       sample.  Populated from AMI ``"kFactor@PMG"``; defaults to
       ``1`` if absent.
   * - ``ecmEnergy``
     - Centre-of-mass collision energy in GeV (e.g. ``13000.0`` for
       13 TeV Run 2).  From AMI ``"ecmEnergy"`` field.
   * - ``genFiltEff``
     - Generator-level filter efficiency: fraction of events passing
       the generator filter (0–1).  From AMI ``"genFiltEff"`` field.
   * - ``completion``
     - Dataset completion fraction (0–1) as reported by AMI
       (``"completion"`` field).
   * - ``beam_energy``
     - Individual beam energy in GeV (half of ``ecmEnergy`` for
       symmetric beams).  From AMI ``"beam_energy"`` field.
   * - ``cross_section_nb``
     - Inclusive production cross-section in nanobarns (nb), set from
       ``meta_t.crossSection_mean``.
   * - ``cross_section_pb``
     - Cross-section converted to picobarns: ``cross_section_nb × 1000``.
   * - ``cross_section_fb``
     - Cross-section converted to femtobarns: ``cross_section_nb × 10⁶``.
   * - ``campaign_luminosity``
     - Integrated luminosity of the ATLAS campaign this sample belongs
       to (in fb⁻¹), stored in ``meta_t.campaign_luminosity``.
   * - ``sum_of_weights``
     - Sum of MC event weights for normalisation, taken from the first
       ``weights_t`` entry in ``misc`` that has a non-negative
       ``processed_events_weighted`` value.

Numeric Properties (``unsigned int``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - ``dsid``
     - ATLAS Dataset Identifier (DSID): the unique integer that
       identifies the MC sample in the ATLAS production system
       (e.g. ``410470`` for the standard ``ttbar`` TOPQ1 sample).
       Read from ``inputConfig.dsid`` in the ``AnalysisTracking`` JSON,
       or from the third x-axis bin of the sum-of-weights histogram.
   * - ``nFiles``
     - Number of files in the dataset as reported by AMI
       (``"nFiles"`` field).
   * - ``totalEvents``
     - Total number of events in the dataset as reported by AMI
       (``"totalEvents"`` field).
   * - ``datasetNumber``
     - AMI internal dataset number (``"datasetNumber"`` field), which
       may differ from ``dsid`` for some derived datasets.

String Properties
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - ``derivationFormat``
     - ATLAS DAOD derivation framework format used to produce the sample
       (e.g. ``"TOPQ1"``). Extracted from the ``inputConfig.derivationFormat``
       key of the ``AnalysisTracking`` JSON tree in the ROOT file, or from the
       ``-DCMAKE_ANALYSISG_CUDA=OFF`` build configuration if absent.
   * - ``AMITag``
     - AMI production tag string (e.g. ``"e7101_s3681_r13144_p5855"``).
       Read from ``inputConfig.amiTag`` in the ``AnalysisTracking`` JSON; if
       absent, extracted from the second-to-last path component of the first
       input file; falls back to the second-to-last component of
       ``sample_name``.
   * - ``generators``
     - Space-separated list of MC generator names read from the
       ``generators`` branch of the ``AnalysisTracking`` tree
       (e.g. ``"PP8"`` for Powheg+Pythia8). The setter strips ``+``
       characters and replaces them with spaces.
   * - ``identifier``
     - AMI dataset identifier string as returned by the AMI API
       ``get_dataset_info`` call (maps to the ``"identifier"`` field in the
       AMI response).
   * - ``DatasetName``
     - Human-readable dataset name.  Set to the ``logicalDatasetName``
       returned by the AMI ``get_dataset_info`` call and also written into
       ``meta_t.DatasetName``.
   * - ``prodsysStatus``
     - Production system status string from AMI (e.g. ``"done"``).
       Maps directly to the ``"prodsysStatus"`` key in the AMI dataset-info
       response.
   * - ``dataType``
     - Data format type string from AMI (e.g. ``"DAOD"``).  Maps to the
       ``"dataType"`` key in the AMI dataset-info response.
   * - ``version``
     - Dataset version string from AMI (e.g. ``"001"``).  Maps to the
       ``"version"`` key in the AMI dataset-info response.
   * - ``PDF``
     - Parton Distribution Function set name used in the generator
       (e.g. ``"PDF4LHC21"``).  Maps to the ``"PDF"`` key in the AMI
       dataset-info response.
   * - ``AtlasRelease``
     - ATLAS software release tag used when producing the sample
       (e.g. ``"AthDerivation-22.2.96"``).  Maps to the ``"AtlasRelease"``
       key in the AMI dataset-info response.
   * - ``principalPhysicsGroup``
     - ATLAS physics group responsible for the sample (e.g. ``"Top"``).
       Maps to the ``"principalPhysicsGroup"`` key in the AMI response.
   * - ``physicsShort``
     - Short physics-process description label assigned by the producing
       group (e.g. ``"ttbar_Pythia8EvtGen_A14_NNPDF23_AFII"``).  Maps to
       ``"physicsShort"`` in the AMI response.
   * - ``generatorName``
     - Full generator name string from AMI (e.g. ``"Powheg+Pythia8"``).
       Maps to ``"generatorName"`` in the AMI response.
   * - ``geometryVersion``
     - ATLAS detector geometry tag
       (e.g. ``"ATLAS-R3S-2021-03-02-00"``).  Maps to
       ``"geometryVersion"`` in the AMI response.
   * - ``conditionsTag``
     - ATLAS conditions database tag
       (e.g. ``"OFLCOND-MC21-SDR-RUN3-09"``).  Maps to
       ``"conditionsTag"`` in the AMI response.
   * - ``generatorTune``
     - Underlying-event tune name of the MC generator
       (e.g. ``"A14 NNPDF2.3 AFII"``).  Maps to ``"generatorTune"`` in
       the AMI response.
   * - ``amiStatus``
     - AMI dataset lifecycle status (e.g. ``"valid"``).  Maps to
       ``"amiStatus"`` in the AMI response.
   * - ``beamType``
     - Beam collision type (e.g. ``"collisions"``).  Maps to
       ``"beamType"`` in the AMI response.
   * - ``productionStep``
     - AMI production step label (e.g. ``"merge"``).  Maps to
       ``"productionStep"`` in the AMI response.
   * - ``projectName``
     - ATLAS MC production project name (e.g. ``"mc21_13p6TeV"``).
       Maps to ``"projectName"`` in the AMI response.
   * - ``statsAlgorithm``
     - Statistical algorithm used for event counting
       (e.g. ``"EventCount"``).  Maps to ``"statsAlgorithm"`` in the AMI
       response.
   * - ``genFilterNames``
     - Comma-separated list of generator filter names applied to the
       sample (e.g. ``"ttbar_filter"``).  Maps to ``"genFilterNames"``
       in the AMI response.
   * - ``file_type``
     - File format type string (e.g. ``"DAOD_TOPQ1"``).  Maps to
       ``"file_type"`` in the AMI response.
   * - ``sample_name``
     - Full sample path or name stored locally.  Also used as a fallback
       source for ``AMITag`` when the ROOT file lacks an
       ``inputConfig.amiTag`` entry: the second-to-last ``"/"``-separated
       component is parsed as the tag.
   * - ``logicalDatasetName``
     - Full AMI logical dataset name including all tags
       (e.g. ``"mc21_13p6TeV.410470.PhPy8EG_ttbar.deriv.DAOD_TOPQ1.e6337_…"``).
       Maps to ``"logicalDatasetName"`` in the AMI response.
   * - ``campaign``
     - ATLAS MC production campaign identifier (e.g. ``"mc21a"``).
       Extracted at scan time from the x-axis bin labels of the
       sum-of-weights ``TH1F`` cutflow histogram: the first bin label
       containing ``"mc"`` is recorded here, with spaces stripped.

Vector Properties
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Property
     - Type
     - Description
   * - ``keywords``
     - ``std::vector<std::string>``
     - AMI keyword list.
   * - ``weights``
     - ``std::vector<std::string>``
     - Available event-weight names.
   * - ``keyword``
     - ``std::vector<std::string>``
     - Short keyword list.
   * - ``fileGUID``
     - ``std::vector<std::string>``
     - GUID strings for each input file.
   * - ``events``
     - ``std::vector<int>``
     - Number of events per input file.
   * - ``run_number``
     - ``std::vector<int>``
     - Run numbers in the dataset.
   * - ``fileSize``
     - ``std::vector<double>``
     - File sizes in bytes.

Map Properties
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Property
     - Type
     - Description
   * - ``inputrange``
     - ``std::map<int, int>``
     - Maps run number → event count.
   * - ``inputfiles``
     - ``std::map<int, std::string>``
     - Maps file index → file path.
   * - ``LFN``
     - ``std::map<std::string, int>``
     - Logical File Name → file index.
   * - ``misc``
     - ``std::map<std::string, weights_t>``
     - Miscellaneous weight name → ``weights_t`` struct.
   * - ``config``
     - ``std::map<std::string, std::string>``
     - Arbitrary key-value configuration pairs.
