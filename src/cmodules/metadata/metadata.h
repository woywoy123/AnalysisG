#ifndef METADATA_H
#define METADATA_H

#include <iostream>
#include <thread>
#include <vector>
#include <map>

struct Leaf
{
    std::string requested = "";
    std::string matched = "";
    std::string branch_name = "";
    std::string tree_name = "";
    std::string path = "";
};

struct Branch
{
    std::string requested = "";
    std::string matched = "";
    std::string tree_name = "";
    std::vector<Leaf*> leaves = {};
};

struct Tree
{
    unsigned int size = 0;
    std::string requested = "";
    std::string matched = "";
    std::vector<Branch*> branches = {};
    std::vector<Leaf*> leaves = {};
};

struct Collect
{
    std::string tr_requested;
    std::string tr_matched;
    std::string br_requested;
    std::string br_matched;
    std::string lf_requested;
    std::string lf_matched;
    std::string lf_path;
    bool valid;
};


struct ExportMetaData
{
    // basic IO content
    std::string hash;
    std::string original_input;
    std::string original_path;
    std::string original_name;

    // requested content of this root file
    std::vector<std::string> req_trees;
    std::vector<std::string> req_branches;
    std::vector<std::string> req_leaves;

    // Missing requested keys
    std::vector<std::string> mis_trees;
    std::vector<std::string> mis_branches;
    std::vector<std::string> mis_leaves;

    // Interpreted content
    std::map<std::string, Leaf> leaves;
    // --- make branches/trees non pointer

    // AnalysisTracking values
    unsigned int dsid;
    std::string AMITag;
    std::string generators;

    bool isMC;
    std::string derivationFormat;
    std::map<int, int> inputrange;
    std::map<int, std::string> inputfiles;
    std::map<std::string, std::string> config;

    // eventnumber is reserved for a ROOT specific mapping
    int eventNumber;

    // event_index is used as a free parameter
    int event_index;

    // search results
    bool found;
    std::string DatasetName;

    // dataset attributes
    double ecmEnergy;
    double genFiltEff;
    double completion;
    double beam_energy;
    double crossSection;
    double crossSection_mean;
    double totalSize;

    unsigned int nFiles;
    unsigned int run_number;
    unsigned int totalEvents;
    unsigned int datasetNumber;

    std::string identifier;
    std::string prodsysStatus;
    std::string dataType;
    std::string version;
    std::string PDF;
    std::string AtlasRelease;
    std::string principalPhysicsGroup;
    std::string physicsShort;
    std::string generatorName;
    std::string geometryVersion;
    std::string conditionsTag;
    std::string generatorTune;
    std::string amiStatus;
    std::string beamType;
    std::string productionStep;
    std::string projectName;
    std::string statsAlgorithm;
    std::string genFilterNames;
    std::string file_type;

    std::vector<std::string> keywords;
    std::vector<std::string> weights;
    std::vector<std::string> keyword;

    std::map<std::string, int> LFN; // logical file name index scheme
    std::vector<std::string> fileGUID;
    std::vector<int> events;
    std::vector<double> fileSize;
};

namespace SampleTracer
{
    class CyMetaData
    {
        public:
            CyMetaData();
            ~CyMetaData();
            void hashing();
            void FindMissingKeys();
            void addconfig(std::string key, std::string val);
            void addsamples(int index, int range, std::string sample);
            void processkeys(std::vector<std::string> keys, unsigned int num_entries);
            ExportMetaData MakeMapping();
            void ImportMetaData(ExportMetaData meta); 

            std::map<std::string, int> GetLength();
            std::map<std::string, std::vector<std::string>> MakeGetter();

            std::string IndexToSample(int index);

            // basic IO content
            std::string hash = "";
            std::string original_input = "";
            std::string original_path = "";
            std::string original_name = "";
            unsigned int chunks = 10;

            // requested content of this root file
            std::vector<std::string> req_trees = {};
            std::vector<std::string> req_branches = {};
            std::vector<std::string> req_leaves = {};

            // Missing requested keys
            std::vector<std::string> mis_trees = {};
            std::vector<std::string> mis_branches = {};
            std::vector<std::string> mis_leaves = {};

            // Interpreted content
            std::map<std::string, Tree> trees = {};
            std::map<std::string, Branch> branches = {};
            std::map<std::string, Leaf> leaves = {};

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
            int eventNumber = -1;

            // event_index is used as a free parameter
            int event_index = 0;

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

            std::vector<std::string> keywords = {};
            std::vector<std::string> weights = {};
            std::vector<std::string> keyword = {};

            std::map<std::string, int> LFN = {}; // logical file name index scheme
            std::vector<std::string> fileGUID = {};
            std::vector<int> events = {};
            std::vector<double> fileSize = {};
    };
}

#endif
