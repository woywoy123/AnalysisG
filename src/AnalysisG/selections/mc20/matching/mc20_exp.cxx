#include "matching.h"
#include <exp_mc20/event.h>

void matching::experimental(event_template* ev){
    exp_mc20* evn = (exp_mc20*)ev; 
    this -> output.push_back(packet_t()); 
    packet_t* pkl = &this -> output[0]; 

    std::vector<particle_template*> tops = evn -> Tops; 
    std::vector<particle_template*> phys_jets = evn -> Jets; 
    std::vector<particle_template*> phys_truth = evn -> PhysicsTruth; 
    std::vector<particle_template*> phys_detector = evn -> Detector; 

    for (size_t x(0); x < tops.size(); ++x){
        top* tpx = (top*)tops[x]; 
        particle* drv = new particle(tpx, true); 
        drv -> root_hash = tpx -> hash; 
        pkl -> truth_tops.push_back(drv);

        std::map<std::string, particle_template*> chx = tpx -> children; 
        std::vector<particle_template*> ch = this -> vectorize(&chx); 
        if (!ch.size()){continue;}

        // ----------- matching children ---------- // 
        std::vector<particle_template*> nu = {}, lep = {}; 
        for (size_t y(0); y < ch.size(); ++y){
            if (ch[y] -> is_nu  &&  nu.size() == 0){ nu.push_back(ch[y]);}
            if (ch[y] -> is_lep && lep.size() == 0){lep.push_back(ch[y]);}
        }
        this -> collect(&ch, &pkl -> children_tops, tpx -> hash); 

        // ---------- matching truth jets -------- //
        std::vector<particle_template*> tjets = {}; 
        for (size_t y(0); y < phys_truth.size(); ++y){
            std::map<std::string, particle_template*> prnt = phys_truth[y] -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}
            tjets.push_back(phys_truth[y]); 
        }

        if (tjets.size()){
            merge_data(&tjets, &nu);
            this -> collect(&tjets, &pkl -> truth_jets, tpx -> hash);
        }

        // ---------- matching jets truth children -------- //
        std::vector<particle_template*> _jets = {}; 
        for (size_t y(0); y < phys_jets.size(); ++y){
            std::map<std::string, particle_template*> prnt = phys_jets[y] -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}
            _jets.push_back(phys_jets[y]); 
        }
        if (!_jets.size()){continue;}

        if (_jets.size()){
            merge_data(&_jets, &nu); 
            merge_data(&_jets, &lep); 
            this -> collect(&_jets, &pkl -> jets_children, tpx -> hash);  
        }

        // ---------- matching jets leptons -------- //
        std::vector<particle_template*> jets_lepton = {}; 
        for (size_t y(0); y < phys_detector.size(); ++y){
            std::map<std::string, particle_template*> prnt = phys_detector[y] -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}
            jets_lepton.push_back(phys_detector[y]); 
        }

        if (jets_lepton.size()){
            merge_data(&jets_lepton, &nu); 
            this -> collect(&jets_lepton, &pkl -> jets_leptons, tpx -> hash);  
        }
    }
}

