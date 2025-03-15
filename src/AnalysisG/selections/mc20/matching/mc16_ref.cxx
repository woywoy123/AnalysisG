#include "matching.h"
#include <bsm_4tops/event.h>

void matching::reference(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 

    this -> output.push_back(packet_t()); 
    packet_t* pkl = &this -> output[0]; 

    std::vector<particle_template*> dleps = {}; 
    merge_data(&dleps, &evn -> Electrons); 
    merge_data(&dleps, &evn -> Muons); 

    std::vector<particle_template*> dtops = evn -> Tops; 
    for (size_t x(0); x < dtops.size(); ++x){

        top* top_i = (top*)dtops[x]; 
        particle* drv = new particle(top_i, true); 
        drv -> root_hash = top_i -> hash;  
        pkl -> truth_tops.push_back(drv); 

        std::vector<particle_template*> ch = this -> vectorize(&top_i -> children); 
        if (!ch.size()){continue;}
      
        // ----------- matching children ---------- // 
        std::vector<particle_template*> ch_ = {}; 
        for (size_t c(0); c < ch.size(); ++c){
            if (!ch[c] -> is_lep && !ch[c] -> is_nu){continue;}
            ch_.push_back(ch[c]); 
        } 
        this -> collect(&ch_, &pkl -> children_tops, top_i -> hash); 

        // ---------- matching truth jets -------- //
        std::vector<particle_template*> tj = {}; 
        this -> downcast(&top_i -> TruthJets, &tj); 
        merge_data(&tj, &ch_); 
        if (top_i -> TruthJets.size()){this -> collect(&tj, &pkl -> truth_jets, top_i -> hash);}
        if (!top_i -> Jets.size()){continue;}

        // ---------- matching jets truth children -------- //
        std::vector<particle_template*> jt = {}; 
        this -> downcast(&top_i -> Jets, &jt);
        merge_data(&jt, &ch_);
        this -> collect(&jt, &pkl -> jets_children, top_i -> hash);  

        // ---------- matching jets leptons -------- //
        std::vector<particle_template*> jts = {}; 
        this -> downcast(&top_i -> Jets, &jts);
        for (size_t c(0); c < dleps.size(); ++c){
            std::map<std::string, particle_template*> pr = dleps[c] -> parents; 
            bool lep_match = false; 
            for (size_t x(0); x < ch.size(); ++x){
                if (!pr.count(ch[x] -> hash)){continue;}
                if (!ch[x] -> is_lep){continue;}
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
        this -> collect(&jts, &pkl -> jets_leptons, top_i -> hash);  
    }
}
