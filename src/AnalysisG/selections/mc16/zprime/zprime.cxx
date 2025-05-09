#include "zprime.h"

zprime::zprime(){this -> name = "zprime";}
zprime::~zprime(){}

selection_template* zprime::clone(){
    return (selection_template*)new zprime();
}

void zprime::merge(selection_template* sl){
    zprime* slt = (zprime*)sl; 
    this -> write(&slt -> zprime_truth_tops, "zprime_truth_tops"); 
    this -> write(&slt -> zprime_children,   "zprime_children");      
    this -> write(&slt -> zprime_truthjets,  "zprime_truthjets");     
    this -> write(&slt -> zprime_jets,       "zprime_jets");          
}

bool zprime::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 

    std::vector<particle_template*> res_tops = {};  
    for (size_t x(0); x < evn -> Tops.size(); ++x){
        top* tp = (top*)evn -> Tops[x]; 
        if (!tp -> from_res){continue;}
        res_tops.push_back(evn -> Tops[x]); 
    }
    return res_tops.size() == 2;
}

bool zprime::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
  
    std::vector<particle_template*> res_tops = {};  
    std::vector<particle_template*> res_children = {}; 
    std::vector<particle_template*> res_truthjets = {}; 
    std::vector<particle_template*> res_jets = {}; 


    std::vector<particle_template*> tmp;
    for (size_t x(0); x < evn -> Tops.size(); ++x){
        top* tp = (top*)evn -> Tops[x]; 
        if (!tp -> from_res){continue;}
        res_tops.push_back(evn -> Tops[x]); 

        tmp = this -> vectorize(&tp -> children); 
        merge_data(&res_children, &tmp); 

        tmp = {};
        this -> downcast(&tp -> TruthJets, &tmp); 
        merge_data(&res_truthjets, &tmp); 

        tmp = {}; 
        this -> downcast(&tp -> Jets, &tmp); 
        merge_data(&res_jets, &tmp); 

        std::vector<particle_template*> lepton = {}; 
        std::vector<particle_template*> neutrino = {}; 

        std::vector<particle_template*> ch_ = this -> vectorize(&tp -> children); 
        for (size_t t(0); t < ch_.size(); ++t){
            if (ch_[t] -> is_nu){neutrino.push_back(ch_[t]);}
            else if (ch_[t] -> is_lep){lepton.push_back(ch_[t]);}
        }
        merge_data(&res_truthjets, &lepton); 
        merge_data(&res_truthjets, &neutrino); 

        for (size_t t(0); t < lepton.size(); ++t){
            std::vector<particle_template*> detector = this -> vectorize(&lepton[t] -> children);
            merge_data(&res_jets, &detector); 
        }
        merge_data(&res_jets, &neutrino); 
    }
    
    particle_template* res_tt = nullptr; 
    this -> sum(&res_tops, &res_tt);
    if (!res_tt){return false;}
    this -> zprime_truth_tops.push_back(res_tt -> mass / 1000); 

    this -> zprime_children.push_back(this -> sum(&res_children)); 
    this -> zprime_truthjets.push_back(this -> sum(&res_truthjets)); 
    this -> zprime_jets.push_back(this -> sum(&res_jets));  
    return true; 
}

