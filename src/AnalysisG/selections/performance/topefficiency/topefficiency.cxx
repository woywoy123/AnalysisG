#include "topefficiency.h"

topefficiency::topefficiency(){this -> name = "topefficiency";}
topefficiency::~topefficiency(){}

selection_template* topefficiency::clone(){
    return (selection_template*)new topefficiency();
}

void topefficiency::merge(selection_template* sl){
    topefficiency* slt = (topefficiency*)sl; 

    merge_data(&this -> predicted_topmass,            &slt -> predicted_topmass); 
    merge_data(&this -> truth_topmass,                &slt -> truth_topmass); 

    merge_data(&this -> predicted_zprime_mass,        &slt -> predicted_zprime_mass); 
    merge_data(&this -> truth_zprime_mass,            &slt -> truth_zprime_mass); 

    merge_data(&this -> n_tops_predictions,           &slt -> n_tops_predictions); 
    merge_data(&this -> n_tops_real,                  &slt -> n_tops_real); 

    sum_data(&this -> truth_res_edge,                 &slt -> truth_res_edge); 
    sum_data(&this -> truth_top_edge,                 &slt -> truth_top_edge);      

    sum_data(&this -> truth_ntops,                    &slt -> truth_ntops);                     
    sum_data(&this -> truth_signal,                   &slt -> truth_signal);                    

    sum_data(&this -> pred_res_edge_score,            &slt -> pred_res_edge_score);             
    sum_data(&this -> pred_top_edge_score,            &slt -> pred_top_edge_score);             

    sum_data(&this -> pred_ntops_score,               &slt -> pred_ntops_score);                
    sum_data(&this -> pred_signal_score,              &slt -> pred_signal_score);               
}

bool topefficiency::selection(event_template* ev){return true;}

int topefficiency::iters(double start, double end, double step){
    return (end - start)/step; 
} 

std::string topefficiency::region(double pt_p, double eta_p){
    int n_pt  = this -> iters(this -> pt_start , this -> pt_end , this -> pt_step); 
    int n_eta = this -> iters(this -> eta_start, this -> eta_end, this -> eta_step); 

    double pt_s_ = 0, pt_e_ = 0; 
    for (int i(0); i < n_pt; ++i){
        pt_s_ = this -> pt_start +    i  * this -> pt_step; 
        pt_e_ = this -> pt_start + (i+1) * this -> pt_step;  
        if (pt_s_ < pt_p && pt_p < pt_e_){break;}
    }

    double eta_s_ = 0, eta_e_ = 0; 
    for (int i(0); i < n_eta; ++i){
        eta_s_ = this -> eta_start +    i  * this -> eta_step; 
        eta_e_ = this -> eta_start + (i+1) * this -> eta_step;  
        if (eta_s_ < eta_p && eta_p < eta_e_){break;}
    }

    std::string key = ""; 
    key += this -> to_string(pt_s_, 0)  + " < $pt_{top}$ < "  + this -> to_string(pt_e_, 0) + "|"; 
    key += this -> to_string(eta_s_, 1) + " < $eta_{top}$ < " + this -> to_string(eta_e_, 1); 
    return key; 
}

bool topefficiency::strategy(event_template* ev){
    gnn_event* evn = (gnn_event*)ev; 
    std::vector<top*> r_tops = evn -> r_tops;  

    std::vector<std::string> keys = {}; 
    for (size_t x(0); x < r_tops.size(); ++x){
        top* top_ = r_tops[x]; 
        std::string key = this -> region(top_ -> pt / 1000, std::abs(top_ -> eta));
        float mass = top_ -> mass / 1000; 
        this -> predicted_topmass[key].push_back(mass);
        if (!this -> n_tops_predictions[key].size()){this -> n_tops_predictions[key].push_back(0);}
        this -> n_tops_predictions[key][0]+=1;
        keys.push_back(key); 
    }

    std::vector<top*> truth_tops = evn -> t_tops;  
    for (size_t x(0); x < truth_tops.size(); ++x){
        top* top_ = truth_tops[x]; 
        std::string key = this -> region(top_ -> pt / 1000, std::abs(top_ -> eta));
        float mass = top_ -> mass / 1000; 
        this -> truth_topmass[key].push_back(mass);
        if (!this -> n_tops_real[key].size()){this -> n_tops_real[key].push_back(0);}
        this -> n_tops_real[key][0]+=1;
        keys.push_back(key); 
    }

    for (size_t x(0); x < keys.size(); ++x){
        std::string key = keys[x]; 
        if (!this -> n_tops_real.count(key)){this -> n_tops_real[key].push_back(0);}
        if (!this -> n_tops_predictions.count(key)){this -> n_tops_predictions[key].push_back(0);}
    }

    std::vector<zprime*> reco_zprime = evn -> r_zprime;  
    for (size_t x(0); x < reco_zprime.size(); ++x){
        zprime* zp_ = reco_zprime[x]; 
        std::string key = this -> region(zp_ -> pt / 1000, std::abs(zp_ -> eta));
        this -> predicted_zprime_mass[key].push_back(zp_ -> mass / 1000); 
    }

    std::vector<zprime*> truth_zprime = evn -> t_zprime;  
    for (size_t x(0); x < truth_zprime.size(); ++x){
        zprime* zp_ = truth_zprime[x];
        std::string key = this -> region(zp_ -> pt / 1000, std::abs(zp_ -> eta));
        this -> truth_zprime_mass[key].push_back(zp_ -> mass / 1000); 
    }

    this -> truth_top_edge = evn -> t_edge_top; 
    this -> truth_res_edge = evn -> t_edge_res; 
    this -> truth_ntops    = {evn -> t_ntops};    
    this -> truth_signal   = {evn -> t_signal};   

    this -> pred_res_edge_score = {evn -> edge_res_scores}; 
    this -> pred_top_edge_score = {evn -> edge_top_scores}; 
    this -> pred_ntops_score    = {evn -> ntops_scores}; 
    this -> pred_signal_score   = {evn -> signal_scores}; 
    return true; 
}


