#ifndef SETTINGS_STRUCTS_H
#define SETTINGS_STRUCTS_H

#include <string>
#include <vector>

struct settings_t {
    std::string output_path = "./ProjectName"; 
    std::string run_name = ""; 

    // machine learning
    int epochs = 10; 
    int kfolds = 10; 
    int num_examples = 3; 
    float train_size = 50; 
   
    bool training = true;
    bool validation = true;
    bool evaluation = true;
    bool continue_training = false;

    // plotting
    std::string var_pt = "pt";
    std::string var_eta = "eta"; 
    std::string var_phi = "phi";
    std::string var_energy = "energy"; 
    std::vector<std::string> targets = {"top_edge"}; 
    
    int nbins = 400;
    int refresh = 10;
    int max_range = 400; 

    bool debug_mode = false;
    int threads = 10; 

}; 

#endif
