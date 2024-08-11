#include "topefficiency.h"

topefficiency::topefficiency(){this -> name = "topefficiency";}
topefficiency::~topefficiency(){}

selection_template* topefficiency::clone(){
    return (selection_template*)new topefficiency();
}

void topefficiency::merge(selection_template* sl){
    topefficiency* slt = (topefficiency*)sl; 

    merge_data(&this -> truthchildren_pt_eta_topmass, &slt -> truthchildren_pt_eta_topmass); 
    merge_data(&this -> truthjets_pt_eta_topmass,     &slt -> truthjets_pt_eta_topmass); 
    merge_data(&this -> jets_pt_eta_topmass,          &slt -> jets_pt_eta_topmass);

    merge_data(&this -> predicted_topmass,            &slt -> predicted_topmass); 
    merge_data(&this -> truth_topmass,                &slt -> truth_topmass); 
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
    int n_pt  = this -> iters(this -> pt_start, this -> pt_end, this -> pt_step); 
    int n_eta = this -> iters(this -> eta_start, this -> eta_end, this -> eta_step); 

    double pt_s_, pt_e_; 
    for (int i(0); i < n_pt; ++i){
        pt_s_ = this -> pt_start +    i  * this -> pt_step; 
        pt_e_ = this -> pt_start + (i+1) * this -> pt_step;  
        if (pt_s_ < pt_p && pt_p < pt_e_){break;}
    }

    double eta_s_, eta_e_; 
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

void topefficiency::build_phasespace(bsm_4tops* evn){
    std::vector<particle_template*> tops = evn -> Tops; 
    for (size_t x(0); x < tops.size(); ++x){
        top* top_ = (top*)tops[x]; 
        std::map<std::string, particle_template*> children_ = top_ -> children;
        std::vector<particle_template*> children = this -> vectorize(&children_); 
        std::vector<particle_template*> leps = {}; 
        this -> get_leptonics(children_, &leps); 
        
        double top_pt  = tops[x] -> pt / 1000; 
        double top_eta = tops[x] -> eta; 
        
        std::string key = this -> region(top_pt, top_eta); 
        float top_mass_ch = this -> sum(&children); 
      
        std::vector<particle_template*> tj_ = {}; 
        this -> downcast(&top_ -> TruthJets, &tj_); 
        merge_data(&tj_, &leps); 
        tj_ = make_unique(&tj_); 
        float top_mass_tj = this -> sum(&tj_);

        std::vector<particle_template*> jets_ = {}; 
        this -> downcast(&top_ -> Jets, &jets_); 
        merge_data(&jets_, &leps); 
        jets_ = make_unique(&jets_); 
        float top_mass_jet = this -> sum(&jets_);

        this -> truthchildren_pt_eta_topmass[key].push_back(top_mass_ch); 
        this -> truthjets_pt_eta_topmass[key].push_back(top_mass_tj); 
        this -> jets_pt_eta_topmass[key].push_back(top_mass_jet); 
    }
}

void topefficiency::build_phasespace(gnn_event* evn){
    std::vector<particle_template*> reco_tops = evn -> reco_tops;  

    for (size_t x(0); x < reco_tops.size(); ++x){
        particle_template* top_ = reco_tops[x]; 
        std::string key = this -> region(top_ -> pt / 1000, top_ -> eta);
        this -> predicted_topmass[key].push_back(top_ -> mass / 1000); 
        if (!this -> n_tops_predictions[key].size()){this -> n_tops_predictions[key].push_back(0);}
        this -> n_tops_predictions[key][0]+=1;
    }

    std::vector<particle_template*> truth_tops = evn -> truth_tops;  
    for (size_t x(0); x < truth_tops.size(); ++x){
        particle_template* top_ = truth_tops[x]; 
        std::string key = this -> region(top_ -> pt / 1000, top_ -> eta);
        this -> truth_topmass[key].push_back(top_ -> mass / 1000); 
        if (!this -> n_tops_real[key].size()){this -> n_tops_real[key].push_back(0);}
        this -> n_tops_real[key][0]+=1;
    }
    
    std::vector<particle_template*> reco_zprime = evn -> reco_zprime;  
    for (size_t x(0); x < reco_zprime.size(); ++x){
        particle_template* zp_ = reco_zprime[x]; 
        std::string key = this -> region(zp_ -> pt / 1000, zp_ -> eta);
        this -> predicted_topmass[key].push_back(zp_ -> mass / 1000); 
    }

    std::vector<particle_template*> truth_zprime = evn -> truth_zprime;  
    for (size_t x(0); x < truth_zprime.size(); ++x){
        particle_template* zp_ = truth_zprime[x]; 
        std::string key = this -> region(zp_ -> pt / 1000, zp_ -> eta);
        this -> truth_topmass[key].push_back(zp_ -> mass / 1000); 
    }

    this -> truth_res_edge = evn -> truth_res_edge; 
    this -> truth_top_edge = evn -> truth_top_edge; 
    this -> truth_ntops    = evn -> truth_ntops;    
    this -> truth_signal.insert(this -> truth_signal.end(), evn -> truth_signal.begin(), evn -> truth_signal.end());   

    this -> pred_res_edge_score = {evn -> pred_res_edge_score}; 
    this -> pred_top_edge_score = {evn -> pred_top_edge_score}; 
    this -> pred_ntops_score    = {evn -> pred_ntops_score   }; 
    this -> pred_signal_score   = {evn -> pred_signal_score  }; 
}

bool topefficiency::strategy(event_template* ev){
    if (ev -> name == "bsm_4tops"){this -> build_phasespace((bsm_4tops*)ev);}
    if (ev -> name == "gnn_event"){this -> build_phasespace((gnn_event*)ev);}
    return true; 
}

