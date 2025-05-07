/**
 * @file meta.h
 * @brief Defines metadata structures for physics analysis datasets.
 *
 * This file contains the declaration of the `meta_t` structure and related types
 * that hold metadata information for physics datasets. This includes information such as
 * dataset identifiers, Monte Carlo status, cross-sections, event counts, and other
 * properties needed for proper normalization and identification of physics samples.
 */

#ifndef META_STRUCTS_H
#define META_STRUCTS_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

/**
 * @struct weights_t
 * @brief Structure to hold weight-related information for events.
 *
 * Stores data related to event weights, which are crucial for proper normalization
 * of Monte Carlo samples. This includes the sum of weights and additional weight factors.
 */
struct weights_t {
    int dsid = -1; ///< Dataset ID (DSID), a unique identifier for the dataset. Initialized to -1.
    bool isAFII = false; ///< Flag indicating if this is an AFII dataset. Initialized to false.
    std::string generator = ""; ///< Name of the generator used. Initialized empty.
    std::string ami_tag = ""; ///< AMI tag associated with the dataset. Initialized empty.
    float total_events_weighted = -1; ///< Total weighted events. Initialized to -1.
    float total_events = -1; ///< Total number of events. Initialized to -1.
    float processed_events = -1; ///< Number of processed events. Initialized to -1.
    float processed_events_weighted = -1; ///< Weighted number of processed events. Initialized to -1.
    float processed_events_weighted_squared = -1; ///< Squared weighted number of processed events. Initialized to -1.
    std::map<std::string, float> hist_data = {}; ///< Histogram data as key-value pairs. Initialized empty.
};

/**
 * @struct meta_t
 * @brief Main structure for storing metadata about physics datasets.
 *
 * Contains comprehensive information about physics datasets, including identifiers,
 * Monte Carlo simulation parameters, cross-sections, event counts, and other properties
 * necessary for proper data handling and normalization in physics analyses.
 */
struct meta_t {
    // AnalysisTracking values
    unsigned int dsid = 0; ///< Dataset ID (DSID), a unique identifier for the dataset. Initialized to 0.
    bool isMC = true; ///< Flag indicating if this is Monte Carlo data (true) or real data (false). Initialized to true.

    std::string derivationFormat = ""; ///< Format of the derived data (e.g., "DAOD_PHYS"). Initialized empty.
    std::map<int, std::string> inputfiles = {}; ///< Map of input files, keyed by an index. Initialized empty.
    std::map<std::string, std::string> config = {}; ///< General configuration settings as key-value pairs. Initialized empty.

    std::string AMITag = ""; ///< ATLAS Metadata Interface tag. Initialized empty.
    std::string generators = ""; ///< Information about Monte Carlo generators used. Initialized empty.

    std::map<int, int> inputrange = {}; ///< Range information for input data, possibly event ranges. Initialized empty.

    // eventnumber is reserved for a ROOT specific mapping
    double eventNumber = -1; ///< Number of events, reserved for ROOT mapping. Initialized to -1.

    // event_index is used as a free parameter
    double event_index = -1; ///< Free parameter for event indexing. Initialized to -1.

    // search results
    bool found = false; ///< Flag indicating if metadata was found. Initialized to false.
    std::string DatasetName = ""; ///< Human-readable name of the dataset. Initialized empty.

    // dataset attributes
    double totalSize = 0; ///< Total size of the dataset, potentially in bytes. Initialized to 0.
    double kfactor = 0; ///< Theoretical correction factor. Initialized to 0.
    double ecmEnergy = 0; ///< Center-of-mass energy of the collision, typically in GeV. Initialized to 0.
    double genFiltEff = 0; ///< Generator filter efficiency for MC simulations. Initialized to 0.
    double completion = 0; ///< Processing completion percentage. Initialized to 0.
    double beam_energy = 0; ///< Energy of the colliding beams. Initialized to 0.
    double crossSection = 0; ///< Cross-section of the physics process. Initialized to 0.
    double crossSection_mean = 0; ///< Mean value of the cross-section. Initialized to 0.
    double campaign_luminosity = 0; ///< Integrated luminosity of the data-taking campaign. Initialized to 0.

    unsigned int nFiles = 0; ///< Number of files in the dataset. Initialized to 0.
    unsigned int totalEvents = 0; ///< Total number of events across all files. Initialized to 0.
    unsigned int datasetNumber = 0; ///< Another numerical identifier for the dataset. Initialized to 0.

    std::string identifier = ""; ///< General string identifier for the metadata. Initialized empty.
    std::string prodsysStatus = ""; ///< Production system status. Initialized empty.
    std::string dataType = ""; ///< Type of data (e.g., "mc", "data"). Initialized empty.
    std::string version = ""; ///< Version string. Initialized empty.
    std::string PDF = ""; ///< Parton Distribution Function information. Initialized empty.
    std::string AtlasRelease = ""; ///< ATLAS software release version. Initialized empty.
    std::string principalPhysicsGroup = ""; ///< Main physics group (e.g., "SM", "SUSY"). Initialized empty.
    std::string physicsShort = ""; ///< Short description of the physics process. Initialized empty.
    std::string generatorName = ""; ///< Name of the Monte Carlo generator. Initialized empty.
    std::string geometryVersion = ""; ///< Version of the detector geometry simulation. Initialized empty.
    std::string conditionsTag = ""; ///< Tag for the detector conditions database. Initialized empty.
    std::string generatorTune = ""; ///< Specific tune (parameters) of the MC generator. Initialized empty.
    std::string amiStatus = ""; ///< Status in the AMI catalog. Initialized empty.
    std::string beamType = ""; ///< Type of colliding beams (e.g., "pp", "PbPb"). Initialized empty.
    std::string productionStep = ""; ///< Step in the production chain. Initialized empty.
    std::string projectName = ""; ///< Project name (e.g., "mc16_13TeV"). Initialized empty.
    std::string statsAlgorithm = ""; ///< Algorithm used for statistics. Initialized empty.
    std::string genFilterNames = ""; ///< Names of generator-level filters. Initialized empty.
    std::string file_type = ""; ///< Type of the file (e.g., "ROOT", "HDF5"). Initialized empty.
    std::string sample_name = ""; ///< User-defined name for the sample. Initialized empty.
    std::string logicalDatasetName = ""; ///< Logical Dataset Name as used in grid systems. Initialized empty.
    std::string campaign = ""; ///< Data-taking or simulation campaign identifier. Initialized empty.

    std::vector<std::string> keywords = {}; ///< List of keywords associated with the dataset. Initialized empty.
    std::vector<std::string> weights = {}; ///< List of weight names or definitions. Initialized empty.
    std::vector<std::string> keyword = {}; ///< Alternative or individual keywords. Initialized empty.

    // Local File Name
    std::vector<int> events = {}; ///< List of event counts, possibly per file. Initialized empty.
    std::vector<int> run_number = {}; ///< List of run numbers. Initialized empty.
    std::vector<double> fileSize = {}; ///< List of file sizes. Initialized empty.
    std::vector<std::string> fileGUID = {}; ///< List of Globally Unique Identifiers for files. Initialized empty.
    std::map<std::string, int> LFN = {}; ///< Map of Logical File Names to integers. Initialized empty.
    std::map<std::string, weights_t> misc = {}; ///< Map for miscellaneous weights information. Initialized empty.
};

#endif
