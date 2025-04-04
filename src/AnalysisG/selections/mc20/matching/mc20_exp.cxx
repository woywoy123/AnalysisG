#include "matching.h"
#include <exp_mc20/event.h>

void matching::experimental(event_template* ev){
    exp_mc20* evn = (exp_mc20*)ev; 

    std::vector<particle_template*> tops          = evn -> Tops; 
    std::vector<particle_template*> phys_jets     = evn -> Jets; 
    std::vector<particle_template*> phys_truth    = evn -> PhysicsTruth; 
    std::vector<particle_template*> phys_detector = evn -> Detector; 

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
        bool is_lep_tru = nu.size() > 0; 

        // ----------- matching top ---------- // 
        int num_jets = 0; 
        std::vector<int> num_merged = {}; 
        std::vector<particle_template*> _jets = {}; 
        std::vector<particle_template*> tjets = {}; 
        std::vector<particle_template*> jets_lepton = {}; 

        bool is_lepx = nu.size() > 0; 
        this -> data.top_partons.num_tops += 1; 
        this -> data.top_partons.num_ltop +=  is_lepx;
        this -> data.top_partons.num_htop += !is_lepx;
        this -> data.top_partons.mass.push_back(tpx -> mass / 1000.0); 

        // ----------- matching children ---------- // 
        this -> dump(&this -> data.top_children, &ch, is_lepx, is_lepx); 

        // ---------- matching truth jets -------- //
        is_lepx = this -> match_obj(&phys_truth, &tjets, tpx -> hash, &num_merged, &num_jets, true); 
        if (tjets.size()){
            merge_data(&tjets, &nu);
            merge_data(&tjets, &lep); 
            this -> dump(&this -> data.top_truthjets, &tjets, is_lep_tru, is_lep_tru, &num_jets, &num_merged); 
        }

        // ---------- matching jets truth children -------- //
        is_lepx = this -> match_obj(&phys_jets, &_jets, tpx -> hash, &num_merged, &num_jets, true); 
        if (_jets.size()){
            merge_data(&_jets, &nu); 
            merge_data(&_jets, &lep); 
            this -> dump(&this -> data.top_jets_children, &_jets, nu.size() > 0, is_lep_tru, &num_jets, &num_merged); 
        }

        // ---------- matching jets leptons -------- //
        is_lepx = this -> match_obj(&phys_detector, &jets_lepton, tpx -> hash, &num_merged, &num_jets, false); 
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

