# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport cython 

cdef extern from "<structs/particles.h>":

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

        map[string, bool] children
        map[string, bool] parents

cdef extern from "<structs/meta.h>":

    struct weights_t:
        int dsid
        bool isAFII
        string generator
        string ami_tag
        float total_events_weighted
        float total_events
        float processed_events
        float processed_events_weighted
        float processed_events_weighted_squared
        map[string, float] hist_data

    struct meta_t:
        unsigned int dsid
        bool isMC

        map[int, int] inputrange
        map[int, string] inputfiles
        map[string, string] config

        double eventNumber
        double event_index

        bool found
        string DatasetName

        double ecmEnergy
        double genFiltEff
        double completion
        double beam_energy
        double crossSection
        double crossSection_mean
        double campaign_luminosity
        double totalSize
        double kfactor

        unsigned int nFiles
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
        string sample_name
        string logicalDatasetName
        string AMITag
        string generators
        string derivationFormat
        string campaign

        vector[string] keywords
        vector[string] weights
        vector[string] keyword
        vector[string] fileGUID

        vector[int] events
        vector[int] run_number
        vector[double] fileSize
        map[string, int] LFN
        map[string, weights_t] misc

cdef extern from "<structs/settings.h>":

    struct settings_t:
        string output_path
        string run_name
        string sow_name
        string metacache_path
        bool fetch_meta
        bool pretagevents

        int epochs
        int kfolds
        int num_examples
        int batch_size
        vector[int] kfold
        float train_size

        bool training
        bool validation
        bool evaluation
        bool continue_training
        string training_dataset
        string graph_cache

        string var_pt
        string var_eta
        string var_phi
        string var_energy
        vector[string] targets

        int nbins
        int refresh
        int max_range

        bool build_cache
        bool debug_mode
        bool selection_root
        int threads

# ------------------- (7.) Add the enum --------------- #

cdef extern from "<structs/element.h>":
    enum data_enum:
        vvf "data_enum::vvf"
        vvd "data_enum::vvd"
        vvl "data_enum::vvl"
        vvi "data_enum::vvi"
        vvb "data_enum::vvb"

        vf  "data_enum::vf"
        vd  "data_enum::vd"
        vl  "data_enum::vl"
        vi  "data_enum::vi"
        vc  "data_enum::vc"
        vb  "data_enum::vb"

        f   "data_enum::f"
        d   "data_enum::d"
        l   "data_enum::l"
        i   "data_enum::i"
        b   "data_enum::b"
        ull "data_enum::ull"
        ui  "data_enum::ui"
        c   "data_enum::c"

    cdef cppclass data_t:
        string leaf_name
        string branch_name
        string tree_name
        string leaf_type
        string* fname
        string path
        data_enum type
        int file_index
        long index

        void flush() except+
        bool next() except+

# ------------------- (8.) Add the interface --------------- #
        bool element(vector[vector[float ]]* data) except +
        bool element(vector[vector[double]]* data) except +
        bool element(vector[vector[long  ]]* data) except +
        bool element(vector[vector[int   ]]* data) except +
        bool element(vector[vector[bool  ]]* data) except +

        bool element(vector[float ]* data) except +
        bool element(vector[double]* data) except +
        bool element(vector[long  ]* data) except +
        bool element(vector[int   ]* data) except +
        bool element(vector[char  ]* data) except +
        bool element(vector[bool  ]* data) except +

        bool element(double* data) except +
        bool element(float*  data) except +
        bool element(long*   data) except +
        bool element(int*    data) except +
        bool element(bool*   data) except +
        bool element(char*   data) except +
        bool element(unsigned int* data) except +
        bool element(unsigned long long* data) except +

# ------------------- (9.) Add the switch (structs.pyx). And you are done =) --------------- #
ctypedef fused basic:
    vector[vector[float ]]
    vector[vector[double]]
    vector[vector[long  ]]
    vector[vector[int   ]]
    vector[vector[bool  ]]

    vector[float ]
    vector[double]
    vector[long  ]
    vector[int   ]
    vector[char  ]
    vector[bool  ]

    bool 
    float
    double 
    int 
    long
    unsigned long long
    unsigned int
    char

cdef dict switch_board(data_t* data)
# ------------------------------------------------------------------------------------------ #
