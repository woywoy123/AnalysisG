#include "toptruthjets.h"

toptruthjets::toptruthjets(){this -> name = "toptruthjets";}
toptruthjets::~toptruthjets(){}

selection_template* toptruthjets::clone(){
    return (selection_template*)new toptruthjets();
}

void toptruthjets::merge(selection_template* sl){
    toptruthjets* slt = (toptruthjets*)sl; 
    merge_data(&this -> top_mass            , &slt -> top_mass            );     
    merge_data(&this -> truthjet_partons    , &slt -> truthjet_partons    );     
    merge_data(&this -> truthjets_contribute, &slt -> truthjets_contribute);     
    merge_data(&this -> truthjet_top        , &slt -> truthjet_top        );     
    merge_data(&this -> truthjet_mass       , &slt -> truthjet_mass       );     
    merge_data(&this -> ntops_lost          , &slt -> ntops_lost          );     
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
        for (size_t i(0); i < t -> TruthJets.size(); ++i){
            merge_data(&tops_tj, &t -> TruthJets[i] -> Tops);
        } 
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
            truthjet* tj1_ = t -> TruthJets[tj1]; 
            for (size_t tj2(tj1+1); tj2 < n_tj; ++tj2){
               truthjet* tj2_ = t -> TruthJets[tj2]; 
               this -> truthjet_top[mode]["dr"].push_back(tj1_ -> DeltaR(tj2_));
               this -> truthjet_top[mode]["energy"].push_back(t -> e / 1000);
               this -> truthjet_top[mode]["pt"].push_back(t -> pt / 1000); 
            }

            for (size_t tj2(0); tj2 < evn -> TruthJets.size(); ++tj2){
                truthjet* tj = (truthjet*)evn -> TruthJets[tj2]; 
                if (this -> contains(&t -> TruthJets, tj)){continue;}
                this -> truthjet_top["background"]["dr"].push_back(tj -> DeltaR(t -> TruthJets[tj1])); 
            }

            for (size_t tp(0); tp < tj1_ -> Parton.size(); ++tp){
                truthjetparton* prt = tj1_ -> Parton[tp]; 
                std::string sym = (std::string(prt -> symbol).size()) ? std::string(prt -> symbol) : "null"; 
                this -> truthjet_partons[mode][sym]["dr"].push_back(prt -> DeltaR(tj1_)); 
                this -> truthjet_partons[mode][sym]["top-pt"].push_back(t -> pt / 1000); 
                this -> truthjet_partons[mode][sym]["top-energy"].push_back(t -> e / 1000); 
                this -> truthjet_partons[mode][sym]["parton-pt"].push_back(prt -> pt / 1000); 
                this -> truthjet_partons[mode][sym]["parton-energy"].push_back(prt -> e / 1000);
                this -> truthjet_partons[mode][sym]["truthjet-pt"].push_back(tj1_ -> pt / 1000);  
                this -> truthjet_partons[mode][sym]["truthjet-energy"].push_back(tj1_ -> e / 1000);  
           }
        }
    }

    std::string mode_ = "background"; 
    this -> truthjet_partons[mode_] = {}; 
    for (size_t tj_(0); tj_ < evn -> TruthJets.size(); ++tj_){
        truthjet* tj = (truthjet*)evn -> TruthJets[tj_]; 
        if (tj -> Tops.size()){continue;}
        for (size_t pr_i(0); pr_i < tj -> Parton.size(); ++pr_i){
            truthjetparton* pr = tj -> Parton[pr_i]; 
            std::string sym = (std::string(pr -> symbol).size()) ? std::string(pr -> symbol) : "null"; 
            this -> truthjet_partons[mode_][sym]["dr"].push_back(pr -> DeltaR(tj)); 
            this -> truthjet_partons[mode_][sym]["parton-pt"].push_back(pr -> pt / 1000); 
            this -> truthjet_partons[mode_][sym]["parton-energy"].push_back(pr -> e / 1000); 
            this -> truthjet_partons[mode_][sym]["truthjet-pt"].push_back(tj -> pt / 1000); 
            this -> truthjet_partons[mode_][sym]["truthjet-energy"].push_back(tj -> e / 1000); 
        }
    }

    // ------- Ghost Parton Energy Contribution --------------- //
    for (size_t x(0); x < evn -> TruthJets.size(); ++x){
        truthjet* tj = (truthjet*)evn -> TruthJets[x]; 
        std::vector<truthjetparton*> prts = this -> make_unique(&tj -> Parton); 
        if (!prts.size()){continue;}
        particle_template* sm = nullptr; 
        this -> sum(&prts, &sm); 
        this -> truthjets_contribute[""]["all"]["pt"].push_back(tj -> pt / sm -> pt); 
        this -> truthjets_contribute[""]["all"]["energy"].push_back(tj -> e / sm -> e); 
        this -> truthjets_contribute[""]["all"]["n-partons"].push_back(float(tj -> Parton.size())); 
        this -> truthjet_mass["all"].push_back(tj -> mass / 1000); 
        this -> truthjet_mass["n-tops"].push_back(float(tj -> Tops.size())); 
        sm = nullptr;

        // This part checks the energy contribution of ghost partons matched to top children
        // and what happens to the reconstructed top-mass when some low energy contributions are removed.
        std::vector<truthjetparton*> alls = {}; 
        std::map<std::string, std::vector<truthjetparton*>> top_maps = {}; 
        for (size_t p(0); p < prts.size(); ++p){
            truthjetparton* prt_ = prts[p]; 
            std::map<std::string, particle_template*> _parents = prt_ -> parents; 
            std::map<std::string, particle_template*>::iterator itr = _parents.begin(); 
            for (; itr != _parents.end(); ++itr){
                std::map<std::string, particle_template*> tops_ = itr -> second -> parents; 
                std::vector<particle_template*> tops = this -> vectorize(&tops_); 
                top_maps[std::string(tops[0] -> hash)].push_back(prt_); 
                alls.push_back(prt_);  
                break; 
            }
        }
        int n_top = int(top_maps.size()); 
        if (!n_top){continue;}
        if (!alls.size()){continue;}

        alls = this -> make_unique(&alls); 
        this -> sum(&alls, &sm); 

        std::map<std::string, std::vector<truthjetparton*>>::iterator itx = top_maps.begin(); 
        for (; itx != top_maps.end(); ++itx){
            particle_template* ps = nullptr; 
            itx -> second = this -> make_unique(&itx -> second); 
            this -> sum(&itx -> second, &ps); 
            this -> truthjets_contribute["n-tops"][std::to_string(n_top)]["energy_r"].push_back(ps -> e / sm -> e); 
        }
    }
    return true; 
}



