#ifndef SETTINGS_STRUCTS_H
#define SETTINGS_STRUCTS_H

#include <string>
#include <vector>

struct settings_t {
    std::string output_path = "./ProjectName"; 
    std::string run_name = ""; 
    std::string sow_name = ""; 
    std::string metacache_path = "./"; 
    bool fetch_meta = false; 
    bool pretagevents = false; 

    // machine learning
    int epochs = 10; 
    int kfolds = 10; 
    int batch_size = 1; 
    std::vector<int> kfold = {}; 

    int num_examples = 3; 
    float train_size = 50; 
   
    bool training = true;
    bool validation = true;
    bool evaluation = true;
    bool continue_training = true;

    std::string training_dataset = ""; 
    std::string graph_cache = ""; 

    // plotting
    std::string var_pt = "pt";
    std::string var_eta = "eta"; 
    std::string var_phi = "phi";
    std::string var_energy = "energy"; 
    std::vector<std::string> targets = {}; 
    
    int nbins = 400;
    int max_range = 400; 
    bool logy = false; 

    int threads = 10; 
    bool debug_mode = false;
    bool build_cache = false; 
    bool selection_root = false; 
}; 

#endif
