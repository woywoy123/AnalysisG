#include "toptruthjets.h"

toptruthjets::toptruthjets(){this -> name = "toptruthjets";}
toptruthjets::~toptruthjets(){}

selection_template* toptruthjets::clone(){
    return (selection_template*)new toptruthjets();
}

void toptruthjets::merge(selection_template* sl){
    toptruthjets* slt = (toptruthjets*)sl; 

    merge_data(&this -> top_mass, &slt -> top_mass); 
    merge_data(&this -> ntops_lost, &slt -> ntops_lost); 
}

bool toptruthjets::selection(event_template* ev){return true;}

bool toptruthjets::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 

    std::vector<particle_template*> tmp; 

    // basic truth top matching to truth jets //
    int lost = 0; 
    for (size_t x(0); x < evn -> Tops.size(); ++x){
        top* t = (top*)evn -> Tops[x]; 
        std::vector<particle_template*> frac = {}; 
        this -> downcast(&t -> TruthJets, &frac);
        if (!frac.size()){++lost; continue;}
        this -> get_leptonics((std::map<std::string, particle_template*>)t -> children, &frac); 
        tmp = this -> make_unique(&frac); 

        // get the top mass of the matched truth jets
        float top_mass = this -> sum(&tmp); 

        // how many truth jets the top is matched to
        int n_tru_j = t -> TruthJets.size(); 
        tmp.clear(); 

        // get how many tops are shared between any matched truth jets.
        std::vector<top*> tops_tj = {}; 
        for (size_t i(0); i < t -> TruthJets.size(); ++i){merge_data(&t -> TruthJets[i] -> Tops, &tops_tj);} 
        this -> downcast(&tops_tj, &tmp); 
        tmp = this -> make_unique(&tmp); 
        int n_tops = tmp.size(); 
        tmp.clear(); 

        // populate the data based on the decay mode
        bool is_lep = t -> lep_decay; 
        std::string mode = (is_lep) ? "leptonic":"hadronic"; 
        this -> top_mass[mode][""][""].push_back(top_mass); 
        this -> top_mass["ntruthjets"][mode][this -> to_string(n_tru_j)].push_back(top_mass);  
        this -> top_mass["merged_tops"][mode][this -> to_string(n_tops)].push_back(top_mass);   
    }
    this -> ntops_lost.push_back(lost); 


    // ------ Kinematic Studies ------ //
    for (size_t x(0); x < evn -> Tops.size(); ++x){
        top* t = (top*)evn -> Tops[x]; 
        std::string mode = (t -> from_res) ? "resonant" : "spectator"; 
        std::string decay = (t -> lep_decay) ? "leptonic" : "hadronic"; 
        mode = mode + "-" + decay; 

        size_t n_tj = t -> TruthJets.size(); 
        for (size_t tj1(0); tj1 < n_tj; ++tj1){
            for (size_t tj2(tj1+1); tj2 < n_tj; ++tj2){
               truthjet* tj1_ = t -> TruthJets[tj1]; 
               truthjet* tj2_ = t -> TruthJets[tj2]; 
               this -> truthjet_top[mode]["dr"].push_back(tj1_ -> DeltaR(tj2_));
               this -> truthjet_top[mode]["energy"].push_back(t -> e / 1000);
               this -> truthjet_top[mode]["pt"].push_back(t -> pt / 1000); 
            }
        }
    }

    return true; 
}


std::vector<particle_template*> toptruthjets::make_unique(std::vector<particle_template*>* inpt){
    std::map<std::string, particle_template*> tmp; 
    for (size_t x(0); x < inpt -> size(); ++x){
        std::string hash = (*inpt)[x] -> hash; 
        tmp[hash] = (*inpt)[x]; 
    } 
   
    std::vector<particle_template*> out = {}; 
    std::map<std::string, particle_template*>::iterator itr; 
    for (itr = tmp.begin(); itr != tmp.end(); ++itr){out.push_back(itr -> second);}
    return out; 
}

