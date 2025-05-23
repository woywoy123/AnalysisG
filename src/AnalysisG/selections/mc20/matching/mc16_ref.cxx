#include "matching.h"
#include <bsm_4tops/event.h>

void matching::reference(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<particle_template*> dleps = {}; 
    merge_data(&dleps, &evn -> Electrons); 
    merge_data(&dleps, &evn -> Muons); 
    std::vector<particle_template*> tops = evn -> Tops; 

    std::map<std::string, std::vector<top*>> jets_; 
    std::map<std::string, std::vector<top*>> tjets_; 
    for (size_t x(0); x < tops.size(); ++x){
        top* tpx = (top*)tops[x]; 
        std::vector<truthjet*> _tjet = tpx -> TruthJets; 
        for (size_t y(0); y < _tjet.size(); ++y){tjets_[_tjet[y] -> hash].push_back(tpx);}

        std::vector<jet*> _jets = tpx -> Jets; 
        for (size_t y(0); y < _jets.size(); ++y){jets_[_jets[y] -> hash].push_back(tpx);}
    }



    for (size_t x(0); x < tops.size(); ++x){
        top* tpx = (top*)tops[x]; 
        std::map<std::string, particle_template*> chx = tpx -> children; 
        std::vector<particle_template*> ch = this -> vectorize(&chx); 
        if (!ch.size()){continue;}
        std::vector<particle_template*> nu = {}; 
        std::vector<particle_template*> lep = {}; 

        for (size_t y(0); y < ch.size(); ++y){
            if (ch[y] -> is_nu  &&  nu.size() == 0){ nu.push_back(ch[y]);}
            if (ch[y] -> is_lep && lep.size() == 0){lep.push_back(ch[y]);}
        }

        // ----------- matching top ---------- // 
        int num_jets = 0; 
        std::vector<int> num_merged = {}; 
        std::vector<particle_template*> _jets = {}; 
        std::vector<particle_template*> tjets = {}; 
        std::vector<particle_template*> jets_lepton = {}; 

        bool is_lepx = nu.size() > 0; 
        bool is_lep_tru = nu.size() > 0; 
        this -> data.top_partons.num_tops += 1; 
        this -> data.top_partons.num_ltop +=  is_lepx;
        this -> data.top_partons.num_htop += !is_lepx;
        this -> data.top_partons.mass.push_back(tpx -> mass / 1000.0); 

        // ----------- matching children ---------- // 
        this -> dump(&this -> data.top_children, &ch, is_lepx, is_lep_tru); 

        // ---------- matching truth jets -------- //
        num_jets = 0; 
        num_merged = {}; 
        for (size_t y(0); y < tpx -> TruthJets.size(); ++y){
            truthjet* ptr = tpx -> TruthJets[y]; 
            num_merged.push_back(tjets_[ptr -> hash].size());  
            tjets.push_back(ptr); 
            num_jets += 1; 
        }

        if (tjets.size()){
            merge_data(&tjets, &nu);
            merge_data(&tjets, &lep); 
            this -> dump(&this -> data.top_truthjets, &tjets, is_lepx, is_lep_tru, &num_jets, &num_merged); 
        }

        // ---------- matching jets truth children -------- //
        num_jets = 0; 
        num_merged = {}; 

        for (size_t y(0); y < tpx -> Jets.size(); ++y){
            jet* ptr = tpx -> Jets[y]; 
            num_merged.push_back(jets_[ptr -> hash].size());  
            _jets.push_back(ptr); 
            jets_lepton.push_back(ptr);
            num_jets += 1; 
        }

        if (_jets.size()){
            merge_data(&_jets, &nu); 
            merge_data(&_jets, &lep); 
            this -> dump(&this -> data.top_jets_children, &_jets, is_lepx, is_lep_tru, &num_jets, &num_merged); 
        }

        is_lepx = false; 
        for (size_t c(0); c < dleps.size(); ++c){
            std::map<std::string, particle_template*> pr = dleps[c] -> parents; 
            bool lep_match = false; 
            for (size_t j(0); j < ch.size(); ++j){
                if (!pr.count(ch[j] -> hash)){continue;}
                if (!ch[j] -> is_lep){continue;}
                lep_match = true;
                break; 
            }
            if (!lep_match){continue;}
            jets_lepton.push_back(dleps[c]); 
            is_lepx = true; 
            break;
        }

        // ---------- matching jets leptons -------- //
        if (!jets_lepton.size()){continue;}
        if (!is_lepx){
            this -> dump(&this -> data.top_jets_leptons, &jets_lepton, is_lepx, is_lep_tru, &num_jets, &num_merged); 
            continue;
        }
        if (jets_lepton.size() < 2){continue;}
        merge_data(&jets_lepton, &nu); 
        this -> dump(&this -> data.top_jets_leptons, &jets_lepton, is_lepx, is_lep_tru, &num_jets, &num_merged); 
    }
}
