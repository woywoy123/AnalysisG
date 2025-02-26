#include "mc20_cur.h"

mc20_current::mc20_current(){this -> name = "mc20_current";}

selection_template* mc20_current::clone(){return (selection_template*)new mc20_current();}

void mc20_current::merge(selection_template* sl){
    mc20_current* slt = (mc20_current*)sl; 
    merge_data(&this -> output, &slt -> output); 
}

bool mc20_current::strategy(event_template* ev){
    ssml_mc20* evn = (ssml_mc20*)ev; 
    std::vector<particle_template*> dleps = evn -> Leptons; 
    std::vector<particle_template*> dtops = evn -> Tops; 
    this -> output.push_back(packet_t()); 
    packet_t* pkl = &this -> output[0]; 

    for (size_t x(0); x < dtops.size(); ++x){
        top* top_i = (top*)dtops[x]; 
        pkl -> truth_tops.push_back(new particle(top_i, true));
        std::map<std::string, particle_template*> children_ = top_i -> children; 
        std::vector<particle_template*> ch = this -> vectorize(&children_); 
        if (!ch.size()){continue;}
       
        bool is_lepton = false;  
        std::vector<particle_template*> ch_ = {}; 
        for (size_t c(0); c < ch.size(); ++c){
            if (!ch[c] -> is_lep && !ch[c] -> is_nu){continue;}
            ch_.push_back(ch[c]); 
            is_lepton = true; 
        } 

        particle_template* ptk = nullptr; 
        if (ch.size()){
            this -> sum(&ch, &ptk); 
            pkl -> children_tops.push_back(new particle(ptk, true)); 
            ptk = nullptr; 
        }

        std::vector<particle_template*> tj = top_i -> truthjets; 
        merge_data(&tj, &ch_); 
        if (top_i -> truthjets.size()){
            this -> sum(&tj, &ptk); 
            pkl -> truth_jets.push_back(new particle(ptk, true));
            ptk = nullptr; 
        }
        if (!top_i -> jets.size()){continue;}
        std::vector<particle_template*> jt = top_i -> jets;
        merge_data(&jt, &ch_);

        this -> sum(&jt, &ptk); 
        pkl -> jets_children.push_back(new particle(ptk, true)); 
        ptk = nullptr; 

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

        this -> sum(&jts, &ptk); 
        pkl -> jets_leptons.push_back(new particle(ptk, true)); 

    }
    return true; 
}

