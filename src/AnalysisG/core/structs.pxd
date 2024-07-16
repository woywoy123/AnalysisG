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

