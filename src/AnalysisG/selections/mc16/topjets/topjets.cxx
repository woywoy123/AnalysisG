#include "topjets.h"

topjets::topjets(){this -> name = "topjets";}
topjets::~topjets(){}

selection_template* topjets::clone(){
    return (selection_template*)new topjets();
}

void topjets::merge(selection_template* sl){
    topjets* slt = (topjets*)sl; 
    merge_data(&this -> top_mass       , &slt -> top_mass       );     
    merge_data(&this -> jet_partons    , &slt -> jet_partons    );     
    merge_data(&this -> jets_contribute, &slt -> jets_contribute);     
    merge_data(&this -> jet_top        , &slt -> jet_top        );     
    merge_data(&this -> jet_mass       , &slt -> jet_mass       );     
    merge_data(&this -> ntops_lost     , &slt -> ntops_lost     );     
}

bool topjets::selection(event_template* ev){return true;}

bool topjets::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<particle_template*> tmp; 

    // basic truth top matching to truth jets //
    int lost = 0; 
    for (size_t x(0); x < evn -> Tops.size(); ++x){
        top* t = (top*)evn -> Tops[x]; 
        std::vector<particle_template*> frac = {}; 
        this -> downcast(&t -> Jets, &frac);
        if (!frac.size()){++lost; continue;}
        this -> get_leptonics((std::map<std::string, particle_template*>)t -> children, &frac); 
        tmp = this -> make_unique(&frac); 

        // get the top mass of the matched truth jets
        float top_mass = this -> sum(&tmp); 

        // how many truth jets the top is matched to
        int n_jets = t -> Jets.size(); 
        tmp.clear(); 

        // get how many tops are shared between any matched jets.
        std::vector<top*> tops_jets = {}; 
        for (size_t i(0); i < t -> Jets.size(); ++i){merge_data(&tops_jets, &t -> Jets[i] -> Tops);} 
        this -> downcast(&tops_jets, &tmp); 
        tmp = this -> make_unique(&tmp); 
        int n_tops = tmp.size(); 
        tmp.clear(); 

        // populate the data based on the decay mode
        bool is_lep = t -> lep_decay; 
        std::string mode = (is_lep) ? "leptonic":"hadronic"; 
        this -> top_mass[mode][""][""].push_back(top_mass); 
        this -> top_mass["njets"][mode][this -> to_string(n_jets)].push_back(top_mass);  
        this -> top_mass["merged_tops"][mode][this -> to_string(n_tops)].push_back(top_mass);   
    }
    this -> ntops_lost.push_back(lost); 


    // ------ Kinematic Studies ------ //
    for (size_t x(0); x < evn -> Tops.size(); ++x){
        top* t = (top*)evn -> Tops[x]; 
        std::string mode = (t -> from_res) ? "resonant" : "spectator"; 
        std::string decay = (t -> lep_decay) ? "leptonic" : "hadronic"; 
        mode = mode + "-" + decay; 

        size_t n_jts = t -> Jets.size(); 
        for (size_t j1(0); j1 < n_jts; ++j1){
            jet* j1_ = t -> Jets[j1]; 
            for (size_t j2(j1+1); j2 < n_jts; ++j2){
               jet* j2_ = t -> Jets[j2]; 
               this -> jet_top[mode]["dr"].push_back(j1_ -> DeltaR(j2_));
               this -> jet_top[mode]["energy"].push_back(t -> e / 1000);
               this -> jet_top[mode]["pt"].push_back(t -> pt / 1000); 
            }

            for (size_t j2(0); j2 < evn -> Jets.size(); ++j2){
                jet* jt = (jet*)evn -> Jets[j2]; 
                if (this -> contains(&t -> Jets, jt)){continue;}
                this -> jet_top["background"]["dr"].push_back(jt -> DeltaR(t -> Jets[j1])); 
            }

            for (size_t tp(0); tp < j1_ -> Parton.size(); ++tp){
                jetparton* prt = j1_ -> Parton[tp]; 
                std::string sym = (std::string(prt -> symbol).size()) ? std::string(prt -> symbol) : "null"; 
                this -> jet_partons[mode][sym]["dr"].push_back(prt -> DeltaR(j1_)); 
                this -> jet_partons[mode][sym]["top-pt"].push_back(t -> pt / 1000); 
                this -> jet_partons[mode][sym]["top-energy"].push_back(t -> e / 1000); 

                this -> jet_partons[mode][sym]["parton-pt"].push_back(prt -> pt / 1000); 
                this -> jet_partons[mode][sym]["parton-energy"].push_back(prt -> e / 1000);

                this -> jet_partons[mode][sym]["jet-pt"].push_back(j1_ -> pt / 1000);  
                this -> jet_partons[mode][sym]["jet-energy"].push_back(j1_ -> e / 1000);  
           }
        }
    }

    std::string mode_ = "background"; 
    this -> jet_partons[mode_] = {}; 
    for (size_t j_(0); j_ < evn -> Jets.size(); ++j_){
        jet* jt = (jet*)evn -> Jets[j_]; 
        if (jt -> Tops.size()){continue;}
        for (size_t pr_i(0); pr_i < jt -> Parton.size(); ++pr_i){
            jetparton* pr = jt -> Parton[pr_i]; 
            std::string sym = (std::string(pr -> symbol).size()) ? std::string(pr -> symbol) : "null"; 
            this -> jet_partons[mode_][sym]["dr"].push_back(pr -> DeltaR(jt)); 
            this -> jet_partons[mode_][sym]["parton-pt"].push_back(pr -> pt / 1000); 
            this -> jet_partons[mode_][sym]["parton-energy"].push_back(pr -> e / 1000); 
            this -> jet_partons[mode_][sym]["jet-pt"].push_back(jt -> pt / 1000); 
            this -> jet_partons[mode_][sym]["jet-energy"].push_back(jt -> e / 1000); 
        }
    }

    // ------- Ghost Parton Energy Contribution --------------- //
    for (size_t x(0); x < evn -> Jets.size(); ++x){
        jet* jt = (jet*)evn -> Jets[x]; 
        std::vector<jetparton*> prts = this -> make_unique(&jt -> Parton); 
        if (!prts.size()){continue;}
        particle_template* sm = nullptr; 
        this -> sum(&prts, &sm); 
        this -> jets_contribute[""]["all"]["pt"].push_back(jt -> pt / sm -> pt); 
        this -> jets_contribute[""]["all"]["energy"].push_back(jt -> e / sm -> e); 
        this -> jets_contribute[""]["all"]["n-partons"].push_back(float(jt -> Parton.size())); 
        this -> jet_mass["all"].push_back(jt -> mass / 1000); 
        this -> jet_mass["n-tops"].push_back(float(jt -> Tops.size())); 
        sm = nullptr;

        // This part checks the energy contribution of ghost partons matched to top children
        // and what happens to the reconstructed top-mass when some low energy contributions are removed.
        std::vector<jetparton*> alls = {}; 
        std::map<std::string, std::vector<jetparton*>> top_maps = {}; 
        for (size_t p(0); p < prts.size(); ++p){
            jetparton* prt_ = prts[p]; 
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

        std::map<std::string, std::vector<jetparton*>>::iterator itx = top_maps.begin(); 
        for (; itx != top_maps.end(); ++itx){
            particle_template* ps = nullptr; 
            itx -> second = this -> make_unique(&itx -> second); 
            this -> sum(&itx -> second, &ps); 
            this -> jets_contribute["n-tops"][std::to_string(n_top)]["energy_r"].push_back(ps -> e / sm -> e); 
        }
    }

    return true; 
}

