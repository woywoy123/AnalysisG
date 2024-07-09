#include "topmatching.h"

topmatching::topmatching(){this -> name = "top-kinematics";}
topmatching::~topmatching(){}

selection_template* topmatching::clone(){
    return (selection_template*)new topmatching();
}

void topmatching::merge(selection_template* sl){
    topmatching* slt = (topmatching*)sl; 
    merge_data(&this -> truth_top       , &slt -> truth_top); 
    merge_data(&this -> no_children     , &slt -> no_children); 
    merge_data(&this -> truth_children  , &slt -> truth_children); 
    merge_data(&this -> truth_jets      , &slt -> truth_jets);  
    merge_data(&this -> n_truth_jets_lep, &slt -> n_truth_jets_lep);  
    merge_data(&this -> n_truth_jets_had, &slt -> n_truth_jets_had);  
    merge_data(&this -> jets_truth_leps , &slt -> jets_truth_leps);  
    merge_data(&this -> jet_leps        , &slt -> jet_leps);         
    merge_data(&this -> n_jets_lep      , &slt -> n_jets_lep);       
    merge_data(&this -> n_jets_had      , &slt -> n_jets_had);
}

bool topmatching::selection(event_template* ev){return true;}

float topmatching::sum(std::vector<particle_template*>* ch){
    particle_template* prt = new particle_template(); 
    for (size_t x(0); x < ch -> size(); ++x){prt -> iadd(ch -> at(x));}
    float mass = prt -> mass / 1000; 
    delete prt; 
    return mass; 
}


bool topmatching::strategy(event_template* ev){


    bsm_4tops* evn = (bsm_4tops*)ev; 

    std::vector<particle_template*> dleps = {}; 
    merge_data(&dleps, &evn -> Electrons); 
    merge_data(&dleps, &evn -> Muons); 

    std::vector<particle_template*> dtops = evn -> Tops; 
    for (size_t x(0); x < dtops.size(); ++x){
        top* top_i = (top*)dtops[x]; 
        this -> truth_top.push_back(float(top_i -> mass)/1000);

        std::map<std::string, particle_template*> children_ = top_i -> children; 
        std::vector<particle_template*> ch = this -> vectorize(&children_); 
        if (!ch.size()){this -> no_children.push_back(1);}
       
        bool is_lepton = false;  
        std::vector<particle_template*> ch_ = {}; 
        for (size_t c(0); c < ch.size(); ++c){
            if (!ch[c] -> is_lep && !ch[c] -> is_nu){continue;}
            ch_.push_back(ch[c]); 
            is_lepton = true; 
        } 
        if (ch.size()){
            this -> truth_children["all"].push_back(this -> sum(&ch)); 
            if (is_lepton){this -> truth_children["lep"].push_back(this -> sum(&ch));}
            else {this -> truth_children["had"].push_back(this -> sum(&ch));} 
        }

        std::vector<particle_template*> tj = this -> downcast(&top_i -> TruthJets); 
        merge_data(&tj, &ch_); 
        if (top_i -> TruthJets.size()){
            float mass = this -> sum(&tj); 
            if (is_lepton){this -> truth_jets["lep"].push_back(mass);}
            else {this -> truth_jets["had"].push_back(mass);}
            this -> truth_jets["all"].push_back(mass);
                
            // find the number of truth jet contributions
            std::string ntj = std::to_string(top_i -> TruthJets.size()) + " - Truth Jets";
            if (is_lepton){this -> n_truth_jets_lep[ntj].push_back(mass);}
            else {this -> n_truth_jets_had[ntj].push_back(mass);}
        }

        if (!top_i -> Jets.size()){continue;}
        std::vector<particle_template*> jt = this -> downcast(&top_i -> Jets);
        merge_data(&jt, &ch_);

        float mass_j = this -> sum(&jt); 
        if (is_lepton){this -> jets_truth_leps["lep"].push_back(mass_j);}    
        else {this -> jets_truth_leps["had"].push_back(mass_j);}
        this -> jets_truth_leps["all"].push_back(mass_j); 


        std::vector<particle_template*> jts = this -> downcast(&top_i -> Jets);
        for (size_t c(0); c < dleps.size(); ++c){
            std::map<std::string, particle_template*> pr = dleps[c] -> parents; 

            bool lep_match = false; 
            for (size_t x(0); x < ch.size(); ++x){
                if (!pr.count(ch[x] -> hash)){continue;}
                lep_match = true; 
                break; 
            }

            if (!lep_match){continue;}
            jts.push_back(dleps[c]); 
        }

        for (size_t c(0); c < ch_.size(); ++c){
            if (!ch_[c] -> is_nu){continue;}
            jts.push_back(ch_[c]); 
        }

        float mass_j_ = this -> sum(&jts); 
        if (is_lepton){this -> jet_leps["lep"].push_back(mass_j_);}
        else {this -> jet_leps["had"].push_back(mass_j_);}
        this -> jet_leps["all"].push_back(mass_j_); 

        std::string _ntj = std::to_string(top_i -> Jets.size()) + " - Jets";
        if (is_lepton){this -> n_jets_lep[_ntj].push_back(mass_j_);}
        else {this -> n_jets_had[_ntj].push_back(mass_j_);}
    }

    return true; 
}
