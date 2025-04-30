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

cdef extern from "<structs/enums.h>":
    enum data_enum:
        vvv_ull "data_enum::vvv_ull"
        vvv_ui  "data_enum::vvv_ui"
        vvv_d   "data_enum::vvv_d"
        vvv_l   "data_enum::vvv_l"
        vvv_f   "data_enum::vvv_f"
        vvv_i   "data_enum::vvv_i"
        vvv_b   "data_enum::vvv_b"
        vvv_c   "data_enum::vvv_c"
        
        vv_ull  "data_enum::vv_ull"
        vv_ui   "data_enum::vv_ui"
        vv_d    "data_enum::vv_d"
        vv_l    "data_enum::vv_l"
        vv_f    "data_enum::vv_f"
        vv_i    "data_enum::vv_i"
        vv_b    "data_enum::vv_b"
        vv_c    "data_enum::vv_c"
        
        v_ull   "data_enum::v_ull"
        v_ui    "data_enum::v_ui"
        v_d     "data_enum::v_d"
        v_l     "data_enum::v_l"
        v_f     "data_enum::v_f"
        v_i     "data_enum::v_i"
        v_b     "data_enum::v_b"
        v_c     "data_enum::v_c"
        
        ull     "data_enum::ull"
        ui      "data_enum::ui"
        d       "data_enum::d"
        l       "data_enum::l"
        f       "data_enum::f"
        i       "data_enum::i"
        b       "data_enum::b"
        c       "data_enum::c"


    enum opt_enum:
        adam               "opt_enum::adam"
        adagrad            "opt_enum::adagrad"
        adamw              "opt_enum::adamw"
        lbfgs              "opt_enum::lbfgs"
        rmsprop            "opt_enum::rmsprop"
        sgd                "opt_enum::sgd"
        invalid_optimizer  "opt_enum::invalid_optimizer"

    enum mlp_init:                           
        uniform         "mlp_init::uniform" 
        normal          "mlp_init::normal" 
        xavier_normal   "mlp_init::xavier_normal" 
        xavier_uniform  "mlp_init::xavier_uniform" 
        kaiming_uniform "mlp_init::kaiming_uniform" 
        kaiming_normal  "mlp_init::kaiming_normal" 
    
    enum loss_enum:                   
        bce                           "loss_enum::bce"  
        bce_with_logits               "loss_enum::bce_with_logits"  
        cosine_embedding              "loss_enum::cosine_embedding"  
        cross_entropy                 "loss_enum::cross_entropy"  
        ctc                           "loss_enum::ctc"  
        hinge_embedding               "loss_enum::hinge_embedding"  
        huber                         "loss_enum::huber"  
        kl_div                        "loss_enum::kl_div"  
        l1                            "loss_enum::l1"  
        margin_ranking                "loss_enum::margin_ranking"  
        mse                           "loss_enum::mse"  
        multi_label_margin            "loss_enum::multi_label_margin"  
        multi_label_soft_margin       "loss_enum::multi_label_soft_margin"  
        multi_margin                  "loss_enum::multi_margin"  
        nll                           "loss_enum::nll"  
        poisson_nll                   "loss_enum::poisson_nll"  
        smooth_l1                     "loss_enum::smooth_l1"  
        soft_margin                   "loss_enum::soft_margin"  
        triplet_margin                "loss_enum::triplet_margin"  
        triplet_margin_with_distance  "loss_enum::triplet_margin_with_distance"  
        invalid_loss                  "loss_enum::invalid_loss"  

    enum graph_enum:                  
        data_graph   "graph_enum::data_graph"   
        data_node    "graph_enum::data_node"   
        data_edge    "graph_enum::data_edge"   
        truth_graph  "graph_enum::truth_graph"   
        truth_node   "graph_enum::truth_node"   
        truth_edge   "graph_enum::truth_edge"   
        edge_index   "graph_enum::edge_index"   
        weight       "graph_enum::weight"   
        batch_index  "graph_enum::batch_index"   
        batch_events "graph_enum::batch_events"   
        pred_graph   "graph_enum::pred_graph"   
        pred_node    "graph_enum::pred_node"   
        pred_edge    "graph_enum::pred_edge"   
        pred_extra   "graph_enum::pred_extra"   
    
    enum mode_enum:                   
        training   "mode_num::training" 
        validation "mode_num::validation" 
        evaluation "mode_num::evaluation" 
    
    enum particle_enum:               
        index   "particle_enum::index"      
        pdgid   "particle_enum::pdgid"      
        pt      "particle_enum::pt"      
        eta     "particle_enum::eta"      
        phi     "particle_enum::phi"      
        energy  "particle_enum::energy"      
        px      "particle_enum::px"      
        pz      "particle_enum::pz"      
        py      "particle_enum::py"      
        mass    "particle_enum::mass"      
        charge  "particle_enum::charge"      
        is_b    "particle_enum::is_b"      
        is_lep  "particle_enum::is_lep"      
        is_nu   "particle_enum::is_nu"      
        is_add  "particle_enum::is_add"      
        pmc     "particle_enum::pmc"      
        pmu     "particle_enum::pmu"      
                



cdef extern from "<structs/element.h>":
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

        void flush() except+ nogil
        bool next() except+ nogil

# ------------------- (8.) Add the interface --------------- #
        bool element(vector[vector[float ]]* data) except + nogil
        bool element(vector[vector[double]]* data) except + nogil
        bool element(vector[vector[long  ]]* data) except + nogil
        bool element(vector[vector[int   ]]* data) except + nogil
        bool element(vector[vector[bool  ]]* data) except + nogil

        bool element(vector[float ]* data) except + nogil
        bool element(vector[double]* data) except + nogil
        bool element(vector[long  ]* data) except + nogil
        bool element(vector[int   ]* data) except + nogil
        bool element(vector[char  ]* data) except + nogil
        bool element(vector[bool  ]* data) except + nogil

        bool element(double* data) except + nogil
        bool element(float*  data) except + nogil
        bool element(long*   data) except + nogil
        bool element(int*    data) except + nogil
        bool element(bool*   data) except + nogil
        bool element(char*   data) except + nogil
        bool element(unsigned int* data) except + nogil
        bool element(unsigned long long* data) except + nogil

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
