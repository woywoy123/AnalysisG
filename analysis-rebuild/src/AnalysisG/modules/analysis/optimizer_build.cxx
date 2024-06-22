#include <generators/analysis.h>

void analysis::build_model_session(){
    if (!this -> model_sessions.size()){return this -> info("No Models Specified. Skipping.");}
    this -> trainer -> kfolds     = this -> kfolds; 
    this -> trainer -> epochs     = this -> epochs; 
    this -> trainer -> var_pt     = this -> var_pt; 
    this -> trainer -> var_eta    = this -> var_eta;
    this -> trainer -> var_phi    = this -> var_phi;
    this -> trainer -> var_energy = this -> var_energy; 
    this -> trainer -> targets    = this -> targets; 
    this -> trainer -> nbins      = this -> nbins; 
    this -> trainer -> max_range  = this -> max_range; 

    this -> trainer -> import_dataloader(this -> loader);
    this -> trainer -> import_model_sessions(&model_sessions[0]); 
    this -> trainer -> check_model_sessions(this -> num_examples); 
    this -> trainer -> launch_model(); 
}
