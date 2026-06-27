#include "topefficiency.h"

selection_template* topefficiency::clone(){
    return (selection_template*)new topefficiency();
}

bool topefficiency::selection(event_template* ev){return true;}
int topefficiency::iters(double start, double end, double step){return (end - start)/step;} 
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

void topefficiency::check_matching(dump_t* ptx, top* trg, std::vector<top*>* ev_tops){
    double tpx = trg -> px; 
    double tpy = trg -> py; 
    double tpz = trg -> pz; 
    double tms = trg -> mass; 

    long bst = 0; 
    double lowC = -1; 
    for (size_t x(0); x < ev_tops -> size(); ++x){
        double px = ev_tops -> at(x) -> px; 
        double py = ev_tops -> at(x) -> py; 
        double pz = ev_tops -> at(x) -> pz; 
        double ms = ev_tops -> at(x) -> mass; 

        double dx = (px - tpx) * (px - tpx); 
        double dy = (py - tpy) * (py - tpy); 
        double dz = (pz - tpz) * (pz - tpz); 
        double dm = (ms - tms) * (ms - tms); 

        double sm = dx + dy + dz + dm; 
        if (lowC < 0){lowC = sm;}
        if (lowC < sm){continue;}
        lowC = sm; bst = x;  
    }
    if (lowC < 0){return;}
    top* tx = ev_tops -> at(bst); 
    ptx -> transfer(ev_tops -> at(bst)); 
    ptx -> lowC.push_back(lowC); 
    if (!tx -> av_score){return;}
    ptx -> ranks.push_back(tx -> av_score); 
}





