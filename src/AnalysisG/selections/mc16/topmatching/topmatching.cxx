#include "topmatching.h"

topmatching::topmatching(){this -> name = "top-matching";}
topmatching::~topmatching(){}

selection_template* topmatching::clone(){
    return (selection_template*)new topmatching();
}

void topmatching::merge(selection_template* sl){
    topmatching* slt = (topmatching*)sl; 

    this -> write(&slt -> truth_top, "top_mass"); 

    this -> write(&slt -> topchildren_mass    , "topchildren_mass"); 
    this -> write(&slt -> topchildren_leptonic, "topchildren_islep"); 

    this -> write(&slt -> toptruthjets_mass    , "toptruthjets_mass"); 
    this -> write(&slt -> toptruthjets_leptonic, "toptruthjets_islep"); 
    this -> write(&slt -> toptruthjets_njets   , "toptruthjets_njets"); 

    this -> write(&slt -> topjets_children_mass    , "topjets_children_mass"); 
    this -> write(&slt -> topjets_children_leptonic, "topjets_children_islep"); 

    this -> write(&slt -> topjets_leptons_mass    , "topjets_leptons_mass"); 
    this -> write(&slt -> topjets_leptons_leptonic, "topjets_leptons_islep"); 
    this -> write(&slt -> topjets_leptons_pdgid   , "topjets_leptons_pdgid"); 
    this -> write(&slt -> topjets_leptons_njets   , "topjets_leptons_njets"); 
}

bool topmatching::selection(event_template* ev){return true;}

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
            this -> topchildren_mass.push_back(this -> sum(&ch)); 
            this -> topchildren_leptonic.push_back(is_lepton);
        }

        std::vector<particle_template*> tj = this -> downcast(&top_i -> TruthJets); 
        merge_data(&tj, &ch_); 
        if (top_i -> TruthJets.size()){
            this -> toptruthjets_mass.push_back(this -> sum(&tj)); 
            this -> toptruthjets_leptonic.push_back(is_lepton);
            this -> toptruthjets_njets.push_back(int(top_i -> TruthJets.size())); 
        }

        if (!top_i -> Jets.size()){continue;}
        std::vector<particle_template*> jt = this -> downcast(&top_i -> Jets);
        merge_data(&jt, &ch_);
        this -> topjets_children_mass.push_back(this -> sum(&jt)); 
        this -> topjets_children_leptonic.push_back(is_lepton);

        int pdgid = 0; 
        std::vector<particle_template*> jts = this -> downcast(&top_i -> Jets);
        for (size_t c(0); c < dleps.size(); ++c){
            std::map<std::string, particle_template*> pr = dleps[c] -> parents; 
            bool lep_match = false; 
            for (size_t x(0); x < ch.size(); ++x){
                if (!pr.count(ch[x] -> hash)){continue;}
                if (!ch[x] -> is_lep){continue;}
                pdgid = ch[x] -> pdgid; 
                lep_match = true;
                break; 
            }
            if (!lep_match){continue;}
            jts.push_back(dleps[c]); 
            break;
        }

        for (size_t c(0); c < ch_.size(); ++c){
            if (!ch_[c] -> is_nu){continue;}
            jts.push_back(ch_[c]); 
        }

        this -> topjets_leptons_mass.push_back(this -> sum(&jts)); 
        this -> topjets_leptons_leptonic.push_back(is_lepton); 
        this -> topjets_leptons_pdgid.push_back(std::abs(pdgid)); 
        this -> topjets_leptons_njets.push_back(int(top_i -> Jets.size())); 
    }
    return true; 
}
