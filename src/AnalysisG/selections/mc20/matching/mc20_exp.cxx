#include "matching.h"
#include <exp_mc20/event.h>

void matching::experimental(event_template* ev){
    exp_mc20* evn = (exp_mc20*)ev; 

    std::vector<particle_template*> tops = evn -> Tops; 
    std::vector<particle_template*> phys_jets = evn -> Jets; 
    std::vector<particle_template*> phys_truth = evn -> PhysicsTruth; 
    std::vector<particle_template*> phys_detector = evn -> Detector; 

    for (size_t x(0); x < tops.size(); ++x){
        top* tpx = (top*)tops[x]; 
        this -> data.top_partons.num_tops += 1; 
        this -> data.top_partons.mass.push_back(tpx -> mass); 

        std::map<std::string, particle_template*> chx = tpx -> children; 
        std::vector<particle_template*> ch = this -> vectorize(&chx); 
        if (!ch.size()){continue;}

        // ----------- matching children ---------- // 
        std::vector<int> pdgid = {}; 
        std::vector<particle_template*> nu = {}, lep = {}; 
        for (size_t y(0); y < ch.size(); ++y){
            if (ch[y] -> is_nu  &&  nu.size() == 0){ nu.push_back(ch[y]);}
            if (ch[y] -> is_lep && lep.size() == 0){lep.push_back(ch[y]);}
            pdgid.push_back(ch[y] -> pdgid); 
        }

        bool is_lepx = nu.size() > 0; 
        this -> data.top_partons.num_ltop +=  is_lepx;
        this -> data.top_partons.num_htop += !is_lepx;

        this -> data.top_children.pdgid.push_back(pdgid);
        this -> data.top_children.is_leptonic.push_back(int( is_lepx)); 
        this -> data.top_children.is_hadronic.push_back(int(!is_lepx)); 
        this -> data.top_children.mass.push_back(this -> sum(&ch));

        this -> data.top_children.num_tops += 1; 
        this -> data.top_children.num_ltop +=  is_lepx;
        this -> data.top_children.num_htop += !is_lepx;

        // ---------- matching truth jets -------- //
        pdgid = {}; 
        is_lepx = false; 
        int num_jets = 0; 
        std::vector<int> num_merged = {}; 
        std::vector<particle_template*> tjets = {}; 
        for (size_t y(0); y < phys_truth.size(); ++y){
            particle_template* ptr = phys_truth[y]; 
            std::map<std::string, particle_template*> prnt = ptr -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}

            tjets.push_back(ptr);
            pdgid.push_back(ptr -> pdgid); 
            if (ptr -> is_nu || ptr -> is_lep){is_lepx = true; continue;}

            num_jets += 1; 
            num_merged.push_back(int(prnt.size()));  
        }

        if (tjets.size()){
            merge_data(&tjets, &nu);
            
            this -> data.top_truthjets.num_tops += 1; 
            this -> data.top_truthjets.num_ltop += is_lepx; 
            this -> data.top_truthjets.num_htop += !is_lepx; 

            this -> data.top_truthjets.is_leptonic.push_back(int( is_lepx)); 
            this -> data.top_truthjets.is_hadronic.push_back(int(!is_lepx)); 

            this -> data.top_truthjets.num_jets.push_back(num_jets); 
            this -> data.top_truthjets.merged.push_back(num_merged); 

            this -> data.top_truthjets.pdgid.push_back(pdgid); 
            this -> data.top_truthjets.mass.push_back(this -> sum(&tjets)); 
        }

        // ---------- matching jets truth children -------- //
        num_jets = 0; 
        pdgid = {}; 
        num_merged = {}; 
        std::vector<particle_template*> _jets = {}; 
        for (size_t y(0); y < phys_jets.size(); ++y){
            particle_template* ptr = phys_jets[y]; 
            std::map<std::string, particle_template*> prnt = ptr -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}

            pdgid.push_back(ptr -> pdgid); 
            if (ptr -> is_nu + ptr -> is_lep){continue;}
            _jets.push_back(phys_jets[y]); 

            num_jets += 1; 
            num_merged.push_back(int(prnt.size()));  
        }
        if (!_jets.size()){continue;}

        if (_jets.size()){
            merge_data(&_jets, &nu); 
            merge_data(&_jets, &lep); 
            
            is_lepx = nu.size() > 0; 
            this -> data.top_jets_children.num_tops += 1; 
            this -> data.top_jets_children.num_ltop += is_lepx; 
            this -> data.top_jets_children.num_htop += !is_lepx; 

            this -> data.top_jets_children.is_leptonic.push_back(int( is_lepx)); 
            this -> data.top_jets_children.is_hadronic.push_back(int(!is_lepx)); 

            this -> data.top_jets_children.num_jets.push_back(num_jets); 
            this -> data.top_jets_children.merged.push_back(num_merged); 

            this -> data.top_jets_children.pdgid.push_back(pdgid); 
            this -> data.top_jets_children.mass.push_back(this -> sum(&_jets)); 
        }

        // ---------- matching jets leptons -------- //
        pdgid = {}; 
        num_merged = {}; 
        num_jets = 0; 
        is_lepx = false;   
        
        std::vector<particle_template*> jets_lepton = {}; 
        for (size_t y(0); y < phys_detector.size(); ++y){
            particle_template* ptr = phys_detector[y];
            std::map<std::string, particle_template*> prnt = ptr -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}
            jets_lepton.push_back(ptr); 

            pdgid.push_back(ptr -> pdgid); 
            if (ptr -> is_nu + ptr -> is_lep){is_lepx = true; continue;}

            num_jets += 1; 
            num_merged.push_back(int(prnt.size()));  
        }

        if (jets_lepton.size()){
            merge_data(&jets_lepton, &nu); 

            this -> data.top_jets_leptons.num_tops += 1; 
            this -> data.top_jets_leptons.num_ltop += is_lepx; 
            this -> data.top_jets_leptons.num_htop += !is_lepx; 

            this -> data.top_jets_leptons.is_leptonic.push_back(int( is_lepx)); 
            this -> data.top_jets_leptons.is_hadronic.push_back(int(!is_lepx)); 

            this -> data.top_jets_leptons.num_jets.push_back(num_jets); 
            this -> data.top_jets_leptons.merged.push_back(num_merged); 

            this -> data.top_jets_leptons.pdgid.push_back(pdgid); 
            this -> data.top_jets_leptons.mass.push_back(this -> sum(&_jets)); 
        }
    }
}

