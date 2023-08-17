from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "metadata.h" namespace "SampleTracer":
    cdef cppclass CyMetaData:
        CyMetaData() except +
        void hashing() except +
        void addsamples(int index, int _range, string sample) except +
        void addconfig(string key, string val) except +
        void processkeys(vector[string], unsigned int num) except +
        void FindMissingKeys() except +
        map[string, int] GetLength() except +
        map[string, vector[string]] MakeGetter() except +
        string IndexToSample(int index) except +

        string hash
        string original_input
        string original_path
        string original_name

        # requested content
        vector[string] req_trees
        vector[string] req_branches
        vector[string] req_leaves

        # requested content
        vector[string] mis_trees
        vector[string] mis_branches
        vector[string] mis_leaves

        # scraped from input
        bool isMC
        bool found
        string AMITag
        string generators
        string derivationFormat
        string DatasetName
        unsigned int dsid
        unsigned int eventNumber
        map[int, string] inputfiles
        map[string, string] config

        # Dataset attributes
        double ecmEnergy
        double genFiltEff
        double completion
        double beam_energy
        double crossSection
        double crossSection_mean
        double totalSize

        unsigned int nFiles
        unsigned int run_number
        unsigned int totalEvents
        unsigned int datasetNumber

        string identifier
        string prodsysStatus
        string dataType
        string version
        string PDF
        string AtlasRelease
        string principalPhysicsGroup
        string physicsShort
        string generatorName
        string geometryVersion
        string conditionsTag
        string generatorTune
        string amiStatus
        string beamType
        string productionStep
        string projectName
        string statsAlgorithm
        string genFilterNames
        string file_type

        vector[string] keywords
        vector[string] weights
        vector[string] keyword

        map[string, int] LFN
        vector[string] fileGUID
        vector[int] events
        vector[double] fileSize







