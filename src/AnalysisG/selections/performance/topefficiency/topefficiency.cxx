#include "topefficiency.h"

topefficiency::topefficiency(){this -> name = "topefficiency";}
topefficiency::~topefficiency(){}

selection_template* topefficiency::clone(){
    return (selection_template*)new topefficiency();
}

void topefficiency::merge(selection_template* sl){
    topefficiency* slt = (topefficiency*)sl; 

    merge_data(&this -> p_topmass, &slt -> p_topmass); 
    merge_data(&this -> t_topmass, &slt -> t_topmass); 

    merge_data(&this -> p_zmass,   &slt -> p_zmass); 
    merge_data(&this -> t_zmass,   &slt -> t_zmass); 

    merge_data(&this -> prob_tops  , &slt -> prob_tops); 
    merge_data(&this -> prob_zprime, &slt -> prob_zprime); 

    merge_data(&this -> n_tru_tops    , &slt -> n_tru_tops); 
    merge_data(&this -> n_pred_tops   , &slt -> n_pred_tops); 
    merge_data(&this -> n_perfect_tops, &slt -> n_perfect_tops); 

    merge_data(&this -> p_decay_region, &slt -> p_decay_region); 
    merge_data(&this -> t_decay_region, &slt -> t_decay_region); 

    merge_data(&this -> p_nodes, &slt -> p_nodes); 
    merge_data(&this -> t_nodes, &slt -> t_nodes); 

    sum_data(&this -> truth_res_edge,       &slt -> truth_res_edge); 
    sum_data(&this -> truth_top_edge,       &slt -> truth_top_edge);      

    sum_data(&this -> truth_ntops,          &slt -> truth_ntops);                     
    sum_data(&this -> truth_signal,         &slt -> truth_signal);                    

    sum_data(&this -> pred_res_edge_score,  &slt -> pred_res_edge_score);             
    sum_data(&this -> pred_top_edge_score,  &slt -> pred_top_edge_score);             

    sum_data(&this -> pred_ntops_score,     &slt -> pred_ntops_score);                
    sum_data(&this -> pred_signal_score,    &slt -> pred_signal_score);               
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
    key += this -> to_string(pt_s_, 0)  + " < $pt_{top}$ < "  + this -> to_string(pt_e_, 0) + ", "; 
    key += this -> to_string(eta_s_, 1) + " < |$\\eta_{top}$| < " + this -> to_string(eta_e_, 1); 
    return key; 
}

std::string topefficiency::decaymode(std::vector<top*> ev_tops){
    std::string out = ""; 
    std::map<std::string, int> decay_mode; 
    for (size_t x(0); x < ev_tops.size(); ++x){
        decay_mode[(ev_tops[x] -> n_leps) ? "l" : "h"] += 1;
    }

    std::map<std::string, int>::iterator itx; 
    for (itx = decay_mode.begin(); itx != decay_mode.end(); ++itx){
        for (size_t x(0); x < itx -> second; ++x){out += itx -> first;}
    }
    return out; 
}

bool topefficiency::strategy(event_template* ev){
    gnn_event* evn = (gnn_event*)ev; 
    std::string hash = evn -> hash; 

    // ---------------- Truth Section ---------------- //
    std::map<std::string, bool> t_top_map; 
    std::vector<top*> truth_tops = evn -> t_tops; 
    std::string decay_region_t = this -> decaymode(truth_tops); 
    for (size_t x(0); x < truth_tops.size(); ++x){
        top* top_ = truth_tops[x]; 
        float mass = top_ -> mass / 1000; 
        std::string key = this -> region(top_ -> pt / 1000, std::abs(top_ -> eta));
        this -> t_topmass[key][hash].push_back(mass);
        t_top_map[top_ -> hash] = false; 
        this -> t_nodes[hash][mass] = top_ -> n_nodes; 
        this -> t_decay_region[decay_region_t][hash].push_back(mass); 
    }
    this -> n_tru_tops[hash] = t_top_map.size(); 

    std::vector<zprime*> truth_zprime = evn -> t_zprime;  
    for (size_t x(0); x < truth_zprime.size(); ++x){
        zprime* zp_ = truth_zprime[x];
        std::string key = this -> region(zp_ -> pt / 1000, std::abs(zp_ -> eta));
        this -> t_zmass[key][hash].push_back(zp_ -> mass / 1000); 
    }

    // ----------------- Reconstructed Section ------------------- //
    std::vector<zprime*> reco_zprime = evn -> r_zprime;  
    for (size_t x(0); x < reco_zprime.size(); ++x){
        zprime* zp_ = reco_zprime[x]; 
        std::string key = this -> region(zp_ -> pt / 1000, std::abs(zp_ -> eta));
        this -> p_zmass[key][hash].push_back(zp_ -> mass / 1000); 
        this -> prob_zprime[key][hash].push_back(zp_ -> av_score); 
    }

    std::vector<top*> reco_tops = evn -> r_tops; 
    std::string decay_region_p = this -> decaymode(reco_tops); 
    for (size_t x(0); x < reco_tops.size(); ++x){
        top* top_ = reco_tops[x]; 
        float mass = top_ -> mass / 1000; 
        std::string key = this -> region(top_ -> pt / 1000, std::abs(top_ -> eta));
        this -> p_topmass[key][hash].push_back(mass);
        this -> prob_tops[key][hash].push_back(top_ -> av_score); 
        this -> p_nodes[hash][mass] = top_ -> n_nodes; 
        this -> p_decay_region[decay_region_p][hash].push_back(mass); 
    }

    // ------------------ Efficiency Reconstruction ------------------- //
    for (size_t s(0); s <= int(1.0/this -> score_step); ++s){
        int perf = 0; 
        int reco = 0; 
        float sc = 1-s*this -> score_step; 
        for (size_t x(0); x < reco_tops.size(); ++x){
            top* top_ = reco_tops[x]; 
            std::string id = top_ -> hash; 
            if (top_ -> av_score <= sc){continue;}
            ++reco; 

            if (!t_top_map.count(id)){continue;}
            if (t_top_map[id]){continue;} // prevent double counting
            t_top_map[id] = true; 
            ++perf;
        }
        this -> n_perfect_tops[hash][sc] = perf; 
        this -> n_pred_tops[hash][sc]    = reco; 
        std::map<std::string, bool>::iterator ib = t_top_map.begin(); 
        for (; ib != t_top_map.end(); ++ib){ib -> second = false;}
    }

    this -> truth_top_edge      = evn -> t_edge_top; 
    this -> pred_top_edge_score = {evn -> edge_top_scores}; 

    this -> truth_res_edge      = evn -> t_edge_res; 
    this -> pred_res_edge_score = {evn -> edge_res_scores}; 

    this -> truth_signal        = std::vector<int>({evn -> t_signal});   
    this -> pred_signal_score   = {evn -> signal_scores}; 

    this -> truth_ntops         = std::vector<int>({evn -> t_ntops}); 
    this -> pred_ntops_score    = {evn -> ntops_scores}; 
    return true; 
}


