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

    merge_data(&this -> prob_tops,   &slt -> prob_tops); 
    merge_data(&this -> prob_zprime, &slt -> prob_zprime); 

    merge_data(&this -> ms_cut_perf_tops, &slt -> ms_cut_perf_tops); 
    merge_data(&this -> ms_cut_reco_tops, &slt -> ms_cut_reco_tops); 
    merge_data(&this -> ms_cut_topmass  , &slt -> ms_cut_topmass  ); 

    merge_data(&this -> kin_truth_tops  , &slt -> kin_truth_tops); 
    merge_data(&this -> ms_kin_perf_tops, &slt -> ms_kin_perf_tops); 
    merge_data(&this -> ms_kin_reco_tops, &slt -> ms_kin_reco_tops); 

    merge_data(&this -> n_tru_tops , &slt -> n_tru_tops); 

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
    key += this -> to_string(eta_s_, 1) + " < |$eta_{top}$| < " + this -> to_string(eta_e_, 1); 
    return key; 
}

std::string topefficiency::region(double pt_p){
    int n_pt  = this -> iters(this -> pt_start , this -> pt_end , this -> pt_step); 
    int n_eta = this -> iters(this -> eta_start, this -> eta_end, this -> eta_step); 

    double pt_s_ = 0, pt_e_ = 0; 
    for (int i(0); i < n_pt; ++i){
        pt_s_ = this -> pt_start +    i  * this -> pt_step; 
        pt_e_ = this -> pt_start + (i+1) * this -> pt_step;  
        if (pt_s_ < pt_p && pt_p < pt_e_){break;}
    }
    return this -> to_string(pt_s_, 0)  + " < pt < "  + this -> to_string(pt_e_, 0) + ", "; 
}


std::string topefficiency::decaymode(std::vector<top*> ev_tops){
    std::string out = ""; 
    std::map<std::string, int> decay_mode; 
    for (size_t x(0); x < ev_tops.size(); ++x){decay_mode[(ev_tops[x] -> is_lep) ? "l" : "h"] += 1;}

    std::map<std::string, int>::iterator itx; 
    for (itx = decay_mode.begin(); itx != decay_mode.end(); ++itx){
        for (size_t x(0); x < itx -> second; ++x){out += itx -> first;}
    }
    return out; 
}

void topefficiency::score_mass(
        double score_h, double score_l, double mass_h, double mass_l, 
        gnn_event* evn, int* perf_tops, std::vector<float>* out_tops, 
        std::map<std::string, int>* kin_perf,
        std::map<std::string, int>* kin_reco
){
    std::vector<top*> cut_tops = {}; 
    for (size_t x(0); x < evn -> r_tops.size(); ++x){
        top* top_ = evn -> r_tops[x]; 
        float mass = top_ -> mass / 1000;
        bool mask = (mass >= mass_l)*(mass < mass_h) * (top_ -> av_score >= score_l) * (top_ -> av_score <= score_h); 
        if (!mask){continue;}
        out_tops -> push_back(mass);  
        cut_tops.push_back(top_); 
    }

    int n_perfect_tops = 0; 
    std::map<int, bool> no_double; 
    for (size_t x(0); x < cut_tops.size(); ++x){
        std::map<std::string, particle_template*> ch_t = cut_tops[x] -> children; 
        std::string key_r = this -> region(cut_tops[x] -> pt / 1000); 
        (*kin_reco)[key_r] += 1; 
        for (size_t y(0); y < evn -> t_tops.size(); ++y){
            if (no_double[y]){continue;}
            std::map<std::string, particle_template*> ch = evn -> t_tops[y] -> children; 
            if (!(ch.size() == ch_t.size())){continue;}

            int trg = 0; 
            std::map<std::string, particle_template*>::iterator itc = ch_t.begin(); 
            for (; itc != ch_t.end(); ++itc){trg += (ch.count(itc -> first)) ? 1 : -10000;}
            if (!(trg == ch.size())){continue;}
            std::string key_t = this -> region(evn -> t_tops[y] -> pt / 1000); 

            no_double[y] = true;
            n_perfect_tops++; 
            (*kin_perf)[key_r] += 1; 
        } 
    }
    *perf_tops = n_perfect_tops; 
    


}

bool topefficiency::strategy(event_template* ev){
    gnn_event* evn = (gnn_event*)ev; 
    std::vector<std::string> keys = {}; 

    std::vector<std::string> spl = this -> split(evn -> filename, "/"); 
    std::string fname = spl[spl.size()-2]; 
    float weight = evn -> weight; 

    for (size_t x(0); x < evn -> r_tops.size(); ++x){
        top* top_ = evn -> r_tops[x]; 
        float mass = top_ -> mass / 1000; 
        std::string key = this -> region(top_ -> pt / 1000, std::abs(top_ -> eta));
        this -> p_topmass[key][fname].push_back(mass);
        this -> prob_tops[key][fname].push_back(top_ -> av_score); 
        keys.push_back(key); 
    }

    for (size_t x(0); x < evn -> t_tops.size(); ++x){
        top* top_ = evn -> t_tops[x]; 
        float mass = top_ -> mass / 1000; 
        std::string key = this -> region(top_ -> pt / 1000, std::abs(top_ -> eta));
        this -> t_topmass[key][fname].push_back(mass);
        keys.push_back(key); 

        key = this -> region(top_ -> pt / 1000); 
        if (!this -> kin_truth_tops.count(key)){this -> kin_truth_tops[key][fname].push_back(0);}
        this -> kin_truth_tops[key][fname][0] += 1; 
    }

    std::vector<zprime*> reco_zprime = evn -> r_zprime;  
    for (size_t x(0); x < reco_zprime.size(); ++x){
        zprime* zp_ = reco_zprime[x]; 
        std::string key = this -> region(zp_ -> pt / 1000, std::abs(zp_ -> eta));
        this -> p_zmass[key][fname].push_back(zp_ -> mass / 1000); 
        this -> prob_zprime[key][fname].push_back(zp_ -> av_score); 
    }

    std::vector<zprime*> truth_zprime = evn -> t_zprime;  
    for (size_t x(0); x < truth_zprime.size(); ++x){
        zprime* zp_ = truth_zprime[x];
        std::string key = this -> region(zp_ -> pt / 1000, std::abs(zp_ -> eta));
        this -> t_zmass[key][fname].push_back(zp_ -> mass / 1000); 
    }

    int mass_size  = this -> iters(this -> mass_start , this -> mass_end , this -> mass_step); 
    int score_size = this -> iters(this -> score_start, this -> score_end, this -> score_step); 
    for (size_t x(0); x < mass_size; ++x){
        double mass_l = this -> mass_avg - 0.5 * this -> mass_step*(1+x); 
        double mass_u = this -> mass_avg + 0.5 * this -> mass_step*(1+x); 

        for (size_t y(0); y < score_size; ++y){
            double score_l = this -> score_avg - 0.5 * this -> score_step*(1+y); 
            double score_u = this -> score_avg + 0.5 * this -> score_step*(1+y); 
            std::string key = this -> to_string(mass_u - mass_l, 2) + "-"  + this -> to_string(score_u - score_l, 2); 

            int n_perfect_tops = 0; 
            std::vector<float> cut_t = {}; 
            std::map<std::string, int> kin_perf, kin_reco; 

            this -> score_mass(score_u, score_l, mass_u, mass_l, evn, &n_perfect_tops, &cut_t, &kin_perf, &kin_reco); 
            this -> ms_cut_perf_tops[key][fname].push_back(n_perfect_tops); 
            this -> ms_cut_reco_tops[key][fname].push_back(cut_t.size()); 
            this -> ms_cut_topmass[key][fname] = cut_t; 
            
            std::map<std::string, int>::iterator itx; 
            for (itx = kin_perf.begin(); itx != kin_perf.end(); ++itx){
                this -> ms_kin_perf_tops[key][itx -> first][fname].push_back(itx -> second);
            }

            for (itx = kin_reco.begin(); itx != kin_reco.end(); ++itx){
                this -> ms_kin_reco_tops[key][itx -> first][fname].push_back(itx -> second);
            }
        }
    }

    this -> n_tru_tops[fname].push_back(evn -> t_tops.size()); 

    this -> truth_top_edge = evn -> t_edge_top; 
    this -> truth_res_edge = evn -> t_edge_res; 
    this -> truth_signal   = std::vector<int>({evn -> t_signal});   

    this -> pred_res_edge_score = {evn -> edge_res_scores}; 
    this -> pred_top_edge_score = {evn -> edge_top_scores}; 
    this -> pred_ntops_score    = {evn -> ntops_scores}; 
    this -> pred_signal_score   = {evn -> signal_scores}; 
    return true; 
}


