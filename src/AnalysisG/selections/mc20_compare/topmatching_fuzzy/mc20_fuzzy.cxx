#include "mc20_fuzzy.h"

mc20_fuzzy::mc20_fuzzy(){this -> name = "mc20_fuzzy";}
mc20_fuzzy::~mc20_fuzzy(){}

selection_template* mc20_fuzzy::clone(){
    return (selection_template*)new mc20_fuzzy();
}

void mc20_fuzzy::merge(selection_template* sl){
    mc20_fuzzy* slt = (mc20_fuzzy*)sl; 
    merge_data(&this -> output, &slt -> output); 
}

bool mc20_fuzzy::selection(event_template* ev){return true;}

bool mc20_fuzzy::strategy(event_template* ev){
    exp_mc20* evn = (exp_mc20*)ev; 
    this -> output.push_back(packet_t()); 
    packet_t* pkl = &this -> output[0]; 

    std::vector<particle_template*> tops = evn -> Tops; 
    std::vector<particle_template*> phys_jets = evn -> Jets; 
    std::vector<particle_template*> phys_truth = evn -> PhysicsTruth; 
    std::vector<particle_template*> phys_detector = evn -> Detector; 

    for (size_t x(0); x < tops.size(); ++x){
        top* tpx = (top*)tops[x]; 
        pkl -> truth_tops.push_back(new particle(tpx, true));
        std::vector<particle_template*> ch_ = this -> vectorize(&tpx -> children); 
        if (!ch_.size()){continue;}

        std::vector<particle_template*> nu = {}, lep = {}; 
        for (size_t y(0); y < ch_.size(); ++y){
            if (ch_[y] -> is_nu  &&  nu.size() == 0){ nu.push_back(ch_[y]);}
            if (ch_[y] -> is_lep && lep.size() == 0){lep.push_back(ch_[y]);}
        }

        particle_template* ptr = nullptr; 
        this -> sum(&ch_, &ptr);
        pkl -> children_tops.push_back(new particle(ptr, true)); 

        std::vector<particle_template*> tjets = {}; 
        for (size_t y(0); y < phys_truth.size(); ++y){
            std::map<std::string, particle_template*> prnt = phys_truth[y] -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}
            tjets.push_back(phys_truth[y]); 
        }
        
        if (tjets.size()){
            merge_data(&tjets, &nu); 

            ptr = nullptr; 
            this -> sum(&tjets, &ptr); 
            pkl -> truth_jets.push_back(new particle(ptr, true)); 
        }

        std::vector<particle_template*> _jets = {}; 
        for (size_t y(0); y < phys_jets.size(); ++y){
            std::map<std::string, particle_template*> prnt = phys_jets[y] -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}
            _jets.push_back(phys_jets[y]); 
        }

        if (_jets.size()){
            merge_data(&_jets, &nu); 
            merge_data(&_jets, &lep); 

            ptr = nullptr; 
            this -> sum(&_jets, &ptr);
            pkl -> jets_children.push_back(new particle(ptr, true)); 
        }

        std::vector<particle_template*> jets_lepton = {}; 
        for (size_t y(0); y < phys_detector.size(); ++y){
            std::map<std::string, particle_template*> prnt = phys_detector[y] -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}
            jets_lepton.push_back(phys_detector[y]); 
        }

        if (jets_lepton.size()){
            merge_data(&jets_lepton, &nu); 

            ptr = nullptr; 
            this -> sum(&jets_lepton, &ptr);
            pkl -> jets_leptons.push_back(new particle(ptr, true)); 
        }
    }
    return true; 
}

