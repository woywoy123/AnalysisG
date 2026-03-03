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

``eventNumber``, ``event_index``, ``totalSize``, ``kfactor``, ``ecmEnergy``,
``genFiltEff``, ``completion``, ``beam_energy``, ``cross_section_nb``,
``cross_section_fb``, ``cross_section_pb``, ``campaign_luminosity``,
``sum_of_weights``.

Numeric Properties (``unsigned int``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dsid``, ``nFiles``, ``totalEvents``, ``datasetNumber``.

String Properties
~~~~~~~~~~~~~~~~~

``derivationFormat``, ``AMITag``, ``generators``, ``identifier``,
``DatasetName``, ``prodsysStatus``, ``dataType``, ``version``, ``PDF``,
``AtlasRelease``, ``principalPhysicsGroup``, ``physicsShort``,
``generatorName``, ``geometryVersion``, ``conditionsTag``, ``generatorTune``,
``amiStatus``, ``beamType``, ``productionStep``, ``projectName``,
``statsAlgorithm``, ``genFilterNames``, ``file_type``, ``sample_name``,
``logicalDatasetName``, ``campaign``.

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
