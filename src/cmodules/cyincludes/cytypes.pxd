from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool

cdef extern from "../abstractions/cytypes.h":
    struct code_t:
        vector[string] input_params
        vector[string] co_vars

        map[string, string] param_space
        string function_name
        string class_name
        string source_code
        string object_code
        string hash

        bool is_class
        bool is_function
        bool is_callable
        bool is_initialized
        bool has_param_variable

    struct leaf_t:
        string requested
        string matched
        string branch_name
        string tree_name
        string path

    struct branch_t:
        string requested
        string matched
        string tree_name
        vector[leaf_t] leaves

    struct tree_t:
        unsigned int size
        string requested
        string matched

        vector[branch_t] branches
        vector[leaf_t] leaves

    struct meta_t:
        string hash
        string original_input
        string original_path
        string original_name

        vector[string] req_trees
        vector[string] req_branches
        vector[string] req_leaves

        vector[string] mis_trees
        vector[string] mis_branches
        vector[string] mis_leaves

        unsigned int dsid
        string AMITag
        string generators

        bool isMC
        string derivationFormat
        map[int, int] inputrange
        map[int, string] inputfiles
        map[string, string] config

        int eventNumber

        int event_index

        bool found
        string DatasetName

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


    struct event_t:

        string event_name
        string commit_hash
        bool deprecated

        bool cached

        double weight
        int event_index
        string event_hash
        string event_tagging
        string event_tree
        string event_root
        map[string, string] pickled_data

        bool graph
        bool selection
        bool event

    struct event_T:
        map[string, string] leaves
        map[string, string] branches
        map[string, string] trees
        event_t event
        meta_t  meta

    struct particle_t:

        double e
        double mass

        double px
        double py
        double pz

        double pt
        double eta
        double phi

        bool cartesian
        bool polar

        double charge
        int pdgid
        int index

        string type
        string hash
        string symbol
        vector[int] lepdef
        vector[int] nudef

        map[string, string] pickle_string

    
