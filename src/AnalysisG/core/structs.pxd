# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

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

    struct meta_t:
        unsigned int dsid
        string AMITag
        string generators

        bool isMC
        string derivationFormat
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
        string sample_name
        string logicalDatasetName

        vector[string] keywords
        vector[string] weights
        vector[string] keyword

        map[string, int] LFN
        vector[string] fileGUID
        vector[int] events
        vector[double] fileSize

cdef extern from "<structs/settings.h>":

    struct settings_t:
        string output_path
        string run_name

        int epochs
        int kfolds
        vector[int] kfold
        int num_examples
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

        bool debug_mode
        int threads

cdef extern from "<structs/element.h>":
    enum data_enum:
        vvf "data_enum::vvf"
        vvd "data_enum::vvd"
        vvl "data_enum::vvl"
        vvi "data_enum::vvi"
        vf  "data_enum::vf"
        vl  "data_enum::vl"
        vi  "data_enum::vi"
        vc  "data_enum::vc"
        f   "data_enum::f"
        l   "data_enum::l"
        i   "data_enum::i"
        ull "data_enum::ull"

    cdef cppclass data_t:
        string leaf_name
        string branch_name
        string tree_name
        string leaf_type
        string path
        data_enum type
        int file_index
        long index

        void flush() except+

        bool next(vector[vector[double]]* data) except +
        bool next(vector[vector[float]]* data) except +
        bool next(vector[vector[long ]]* data) except +
        bool next(vector[vector[int  ]]* data) except +

        bool next(vector[float]* data) except +
        bool next(vector[long]* data) except +
        bool next(vector[int]* data) except +
        bool next(vector[char ]* data) except +

        bool next(float* data) except +
        bool next(long* data) except +
        bool next(int* data) except +
        bool next(unsigned long long* data) except +


cdef inline dict switch_board(data_t* data):

    cdef vector[vector[float]] vvf
    if data.type == data_enum.vvf and data.next(&vvf): return {data.path : vvf}

    cdef vector[vector[long]] vvl
    if data.type == data_enum.vvl and data.next(&vvl): return {data.path : vvl}

    cdef vector[vector[int]] vvi
    if data.type == data_enum.vvi and data.next(&vvi): return {data.path : vvi}

    cdef vector[vector[double]] vvd
    if data.type == data_enum.vvd and data.next(&vvd): return {data.path : vvd}

    cdef vector[float] vf
    if data.type == data_enum.vf and data.next(&vf): return {data.path : vf}

    cdef vector[long] vl
    if data.type == data_enum.vl and data.next(&vl): return {data.path : vl}

    cdef vector[int] vi
    if data.type == data_enum.vi and data.next(&vi): return {data.path : vi}

    cdef vector[char] vc
    if data.type == data_enum.vc and data.next(&vc): return {data.path : vc}

    cdef float f
    if data.type == data_enum.f and data.next(&f): return {data.path : f}

    cdef long l
    if data.type == data_enum.l and data.next(&l): return {data.path : l}

    cdef int i
    if data.type == data_enum.i and data.next(&i): return {data.path : i}

    cdef unsigned long long ull
    if data.type == data_enum.ull and data.next(&ull): return {data.path : ull}

    return {}


