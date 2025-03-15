#include "matching.h"
#include <ssml_mc20/event.h>

void matching::current(event_template* ev){
    ssml_mc20* evn = (ssml_mc20*)ev; 
    std::vector<particle_template*> dleps = evn -> Leptons; 
    std::vector<particle_template*> dtops = evn -> Tops; 
    this -> output.push_back(packet_t()); 
    packet_t* pkl = &this -> output[0]; 

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
        this -> collect(&ch, &pkl -> children_tops, top_i -> hash); 

        // ---------- matching truth jets -------- //
        std::vector<particle_template*> tj = top_i -> truthjets; 
        merge_data(&tj, &ch_); 
        if (top_i -> truthjets.size()){this -> collect(&tj, &pkl -> truth_jets, top_i -> hash);}
        if (!top_i -> jets.size()){continue;}
        
        // ---------- matching jets truth children -------- //
        std::vector<particle_template*> jt = top_i -> jets;
        merge_data(&jt, &ch_);
        this -> collect(&jt, &pkl -> jets_children, top_i -> hash);  

        std::vector<particle_template*> jts = top_i -> jets;
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


