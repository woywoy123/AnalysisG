#ifndef META_STRUCTS_H
#define META_STRUCTS_H
#include <iostream>
#include <string>
#include <vector>
#include <map>

struct weights_t {
    int dsid = -1; 
    bool isAFII = false; 
    std::string generator = ""; 
    std::string ami_tag = ""; 
    float total_events_weighted = -1;
    float total_events = -1; 
    float processed_events = -1; 
    float processed_events_weighted = -1; 
    float processed_events_weighted_squared = -1;
    std::map<std::string, float> hist_data = {}; 
}; 

struct meta_t {
    // AnalysisTracking values
    unsigned int dsid = 0;
    bool isMC = true;

    std::string derivationFormat = "";
    std::map<int, std::string> inputfiles = {};
    std::map<std::string, std::string> config = {};

    std::string AMITag = "";
    std::string generators = "";

    std::map<int, int> inputrange = {};

    // eventnumber is reserved for a ROOT specific mapping
    double eventNumber = -1;

    // event_index is used as a free parameter
    double event_index = -1;

    // search results
    bool found = false;
    std::string DatasetName = "";

    // dataset attributes
    double totalSize = 0;
    double kfactor   = 0; 
    double ecmEnergy = 0;
    double genFiltEff = 0;
    double completion = 0;
    double beam_energy = 0;
    double crossSection = 0;
    double crossSection_mean = 0;
    double campaign_luminosity = 0; 

    unsigned int nFiles = 0;
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
    std::string campaign = "";

    std::vector<std::string> keywords = {};
    std::vector<std::string> weights = {};
    std::vector<std::string> keyword = {};

    // Local File Name
    std::vector<int> events = {};
    std::vector<int> run_number = {};
    std::vector<double> fileSize = {};
    std::vector<std::string> fileGUID = {};
    std::map<std::string, int> LFN = {};
    std::map<std::string, weights_t> misc = {}; 
};

#endif
