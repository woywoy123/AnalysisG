.. _meta:

The C++ Interface
^^^^^^^^^^^^^^^^^


.. cpp:class:: meta: public tools, public notification

    .. cpp:function:: meta()

    .. cpp:function:: ~meta()

    .. cpp:function:: const folds_t* get_tags(std::string hash) 

    .. cpp:function:: void scan_data(TObject* obj) 

    .. cpp:function:: void scan_sow(TObject* obj) 

    .. cpp:function:: void parse_json(std::string inpt) 

    .. cpp:function:: std::string hash(std::string fname) 

    .. cpp:var:: rapidjson::Document* rpd = nullptr

    .. cpp:var:: meta_t meta_data 

    .. cpp:var:: std::string metacache_path 

    .. cpp:var:: cproperty<bool, meta> isMC 

    .. cpp:var:: cproperty<bool, meta> found 

    .. cpp:var:: cproperty<double, meta> eventNumber 

    .. cpp:var:: cproperty<double, meta> event_index 

    .. cpp:var:: cproperty<double, meta> totalSize 

    .. cpp:var:: cproperty<double, meta> kfactor  

    .. cpp:var:: cproperty<double, meta> ecmEnergy 

    .. cpp:var:: cproperty<double, meta> genFiltEff 

    .. cpp:var:: cproperty<double, meta> completion 

    .. cpp:var:: cproperty<double, meta> beam_energy 

    .. cpp:var:: cproperty<double, meta> cross_section_nb 

    .. cpp:var:: cproperty<double, meta> cross_section_fb 

    .. cpp:var:: cproperty<double, meta> cross_section_pb 

    .. cpp:var:: cproperty<double, meta> campaign_luminosity  

    .. cpp:var:: cproperty<double, meta> sum_of_weights 

    .. cpp:var:: cproperty<unsigned int, meta> dsid                                    

    .. cpp:var:: cproperty<unsigned int, meta> nFiles 

    .. cpp:var:: cproperty<unsigned int, meta> totalEvents 

    .. cpp:var:: cproperty<unsigned int, meta> datasetNumber

    .. cpp:var:: cproperty<std::string, meta> derivationFormat 

    .. cpp:var:: cproperty<std::string, meta> AMITag 

    .. cpp:var:: cproperty<std::string, meta> generators 

    .. cpp:var:: cproperty<std::string, meta> identifier 

    .. cpp:var:: cproperty<std::string, meta> DatasetName 

    .. cpp:var:: cproperty<std::string, meta> prodsysStatus 

    .. cpp:var:: cproperty<std::string, meta> dataType 

    .. cpp:var:: cproperty<std::string, meta> version 

    .. cpp:var:: cproperty<std::string, meta> PDF 

    .. cpp:var:: cproperty<std::string, meta> AtlasRelease 

    .. cpp:var:: cproperty<std::string, meta> principalPhysicsGroup 

    .. cpp:var:: cproperty<std::string, meta> physicsShort 

    .. cpp:var:: cproperty<std::string, meta> generatorName 

    .. cpp:var:: cproperty<std::string, meta> geometryVersion 

    .. cpp:var:: cproperty<std::string, meta> conditionsTag 

    .. cpp:var:: cproperty<std::string, meta> generatorTune 

    .. cpp:var:: cproperty<std::string, meta> amiStatus 

    .. cpp:var:: cproperty<std::string, meta> beamType 

    .. cpp:var:: cproperty<std::string, meta> productionStep 

    .. cpp:var:: cproperty<std::string, meta> projectName 

    .. cpp:var:: cproperty<std::string, meta> statsAlgorithm 

    .. cpp:var:: cproperty<std::string, meta> genFilterNames 

    .. cpp:var:: cproperty<std::string, meta> file_type 

    .. cpp:var:: cproperty<std::string, meta> sample_name  

    .. cpp:var:: cproperty<std::string, meta> logicalDatasetName  

    .. cpp:var:: cproperty<std::string, meta> campaign 

    .. cpp:var:: cproperty<std::vector<std::string>, meta> keywords 

    .. cpp:var:: cproperty<std::vector<std::string>, meta> weights 

    .. cpp:var:: cproperty<std::vector<std::string>, meta> keyword 

    .. cpp:var:: cproperty<std::vector<std::string>, meta> fileGUID 

    .. cpp:var:: cproperty<std::vector<int>, meta> events 

    .. cpp:var:: cproperty<std::vector<int>, meta> run_number 

    .. cpp:var:: cproperty<std::vector<double>, meta> fileSize 

    .. cpp:var:: cproperty<std::map<int, int>, meta> inputrange 

    .. cpp:var:: cproperty<std::map<int, std::string>, meta> inputfiles 

    .. cpp:var:: cproperty<std::map<std::string, int>, meta> LFN 

    .. cpp:var:: cproperty<std::map<std::string, weights_t>, meta> misc  

    .. cpp:var:: cproperty<std::map<std::string, std::string>, meta> config


The Python Interface
^^^^^^^^^^^^^^^^^^^^

.. py:class:: httpx(pyAMI.httpclient.HttpClient)

   .. py:function::  __init__(self, config)
   
   .. py:function::  connect(self, endpoint)

.. py:class:: atlas(pyAMI.client.Client)

   .. py:function:: __init__()


.. py:class:: MetaLookup

   .. py:function::  __call__(inpt)

   :ivar DatasetName

   :ivar CrossSection

   :ivar ExpectedEvents

   :ivar SumOfWeights

   :ivar GenerateData

.. py:class:: Data

    .. py:function:: __add__(self, Data other)

    .. py:function:: __radd__(self, other)

    :ivar weights

    :ivar data


.. py:class:: Meta

    .. py:function:: __reduce__(self)

    .. py:function:: __str__(self)

    .. py:function:: expected_events(self, float lumi = 140.1)

    .. py:function:: GetSumOfWeights(self, str name)

    .. py:function:: FetchMeta(self, int dsid, str amitag)

    .. py:function:: hash(self, str val)


    :ivar str MetaCachePath

    :ivar float SumOfWeights

    :ivar int dsid

    :ivar int amitag

    :ivar str generators

    :ivar bool isMC

    :ivar str derivationFormat

    :ivar int eventNumber

    :ivar float ecmEnergy 
    
    :ivar float genFiltEff

    :ivar float kfactor
    
    :ivar genFiltEff
   
    :ivar float completion
    
    :ivar completion

    :ivar float beam_energy
    
    :ivar float crossSection
    
    :ivar float crossSection_mean
    
    :ivar float totalSize

    :ivar int nFiles
    
    :ivar int run_number
    
    :ivar int totalEvents

    :ivar totalEvents

    :ivar datasetNumber

    :ivar identifier

    :ivar prodsysStatus

    :ivar dataType

    :ivar version

    :ivar PDF

    :ivar AtlasRelease

    :ivar principalPhysicsGroup

    :ivar physicsShort

    :ivar generatorName

    :ivar geometryVersion

    :ivar conditionsTag

    :ivar generatorTune

    :ivar amiStatus

    :ivar beamType

    :ivar productionStep

    :ivar projectName

    :ivar statsAlgorithm

    :ivar genFilterNames

    :ivar file_type

    :ivar DatasetName

    :ivar logicalDatasetName

    :ivar event_index

    :ivar keywords

    :ivar weights 

    :ivar keyword

    :ivar found   

    :ivar config    

    :ivar Files 

    :ivar fileGUID

    :ivar events    

    :ivar fileSize  

    :ivar sample_name

    :ivar campaign
