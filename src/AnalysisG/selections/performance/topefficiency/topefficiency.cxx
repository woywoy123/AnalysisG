#include "topefficiency.h"

topefficiency::topefficiency(){this -> name = "topefficiency";}
topefficiency::~topefficiency(){}

selection_template* topefficiency::clone(){
    return (selection_template*)new topefficiency();
}

void topefficiency::merge(selection_template* sl){
    topefficiency* slt = (topefficiency*)sl; 

    merge_data(&this -> truthchildren_pt_eta_topmass, &slt -> truthchildren_pt_eta_topmass); 
    merge_data(&this -> truthjets_pt_eta_topmass, &slt -> truthjets_pt_eta_topmass); 
    merge_data(&this -> jets_pt_eta_topmass, &slt -> jets_pt_eta_topmass); 
}

bool topefficiency::selection(event_template* ev){return true;}

bool topefficiency::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<particle_template*> tops = evn -> Tops; 

    double pt_s = 0; 
    double pt_e = 1500; 
    double step_pt = 100; 
    int n_pt = (pt_e - pt_s)/step_pt; 

    double eta_s = -6;
    double eta_e = 6; 
    double step_eta = 0.5; 
    int n_eta = (eta_e - eta_s)/step_eta; 

    for (size_t x(0); x < tops.size(); ++x){
        top* top_ = (top*)tops[x]; 
        std::map<std::string, particle_template*> children_ = top_ -> children;
        std::vector<particle_template*> children = this -> vectorize(&children_); 
        std::vector<particle_template*> leps = {}; 
        this -> get_leptonics(children_, &leps); 
        
        double top_pt = tops[x] -> pt / 1000; 
        double top_eta = tops[x] -> eta; 

        double pt_s_ = -99; 
        double pt_e_ = -99; 
        for (size_t pt(0); pt < n_pt; ++pt){
            pt_s_ = pt_s + pt*step_pt; 
            pt_e_ = pt_s + (pt+1)*step_pt;  
            if (pt_s_ < top_pt && top_pt < pt_e_){break;}
        }

        double eta_s_ = -99; 
        double eta_e_ = -99; 
        for (size_t eta(0); eta < n_eta; ++eta){
            eta_s_ = eta_s + eta*step_eta; 
            eta_e_ = eta_s + (eta+1)*step_eta;  
            if (eta_s_ < top_eta && top_eta < eta_e_){break;}
        }
 
        std::string key = ""; 
        key += std::to_string(pt_s) + "< pt_{top} < " + std::to_string(pt_e) + "|"; 
        key += std::to_string(eta_s) + "< eta_{top} < " + std::to_string(eta_e); 
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
    return true; 
}

