#ifndef META_STRUCTS_H
#define META_STRUCTS_H
#include <iostream>
#include <string>
#include <vector>
#include <map>

struct meta_t {
    // basic IO content
    std::string hash = "";
    std::string original_input = "";
    std::string original_path = "";
    std::string original_name = "";

    // requested content of this root file
    std::vector<std::string> req_trees = {};
    std::vector<std::string> req_branches = {};
    std::vector<std::string> req_leaves = {};

    // Missing requested keys
    std::vector<std::string> mis_trees = {};
    std::vector<std::string> mis_branches = {};
    std::vector<std::string> mis_leaves = {};

    // AnalysisTracking values
    unsigned int dsid = 0;
    std::string AMITag = "";
    std::string generators = "";

    bool isMC = true;
    std::string derivationFormat = "";
    std::map<int, int> inputrange = {};
    std::map<int, std::string> inputfiles = {};
    std::map<std::string, std::string> config = {};

    // eventnumber is reserved for a ROOT specific mapping
    double eventNumber = -1;

    // event_index is used as a free parameter
    double event_index = -1;

    // search results
    bool found = false;
    std::string DatasetName = "";

    // dataset attributes
    double ecmEnergy = 0;
    double genFiltEff = 0;
    double completion = 0;
    double beam_energy = 0;
    double crossSection = 0;
    double crossSection_mean = 0;
    double totalSize = 0;

    unsigned int nFiles = 0;
    unsigned int run_number = 0;
    unsigned int totalEvents = 0;
    unsigned int datasetNumber = 0;

    std::string identifier = "";
    std::string prodsysStatus = "";
    std::string dataType = "";
    std::string version = "";
    std::string PDF = "";
    std::string AtlasRelease = "";
    std::string principalPhysicsGroup = "";
    std::string physicsShort = "";
    std::string generatorName = "";
    std::string geometryVersion = "";
    std::string conditionsTag = "";
    std::string generatorTune = "";
    std::string amiStatus = "";
    std::string beamType = "";
    std::string productionStep = "";
    std::string projectName = "";
    std::string statsAlgorithm = "";
    std::string genFilterNames = "";
    std::string file_type = "";
    std::string sample_name = ""; 
    std::string logicalDatasetName = ""; 

    std::vector<std::string> keywords = {};
    std::vector<std::string> weights = {};
    std::vector<std::string> keyword = {};

    // Local File Name
    std::map<std::string, int> LFN = {};
    std::vector<std::string> fileGUID = {};
    std::vector<int> events = {};
    std::vector<double> fileSize = {};
};

#endif
