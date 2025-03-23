#include "matching.h"
#include <ssml_mc20/event.h>

void matching::current(event_template* ev){
    ssml_mc20* evn = (ssml_mc20*)ev; 
    std::vector<particle_template*> dleps = evn -> Leptons; 
    std::vector<particle_template*> tops = evn -> Tops; 

    for (size_t x(0); x < tops.size(); ++x){
        top* tpx = (top*)tops[x]; 

        std::map<std::string, particle_template*> chx = tpx -> children; 
        std::vector<particle_template*> ch = this -> vectorize(&chx); 
        if (!ch.size()){continue;}
      
        // ----------- matching children ---------- // 
        std::vector<particle_template*> nu = {}, lep = {}; 
        for (size_t y(0); y < ch.size(); ++y){
            if (ch[y] -> is_nu  &&  nu.size() == 0){ nu.push_back(ch[y]);}
            if (ch[y] -> is_lep && lep.size() == 0){lep.push_back(ch[y]);}
        }

        int num_jets = 0; 
        bool is_lepx = nu.size() > 0; 
        std::vector<int> num_merged = {}; 
        this -> data.top_partons.num_tops += 1; 
        this -> data.top_partons.num_ltop +=  is_lepx;
        this -> data.top_partons.num_htop += !is_lepx;
        this -> data.top_partons.mass.push_back(tpx -> mass / 1000.0); 

        // ----------- matching children ---------- // 
        this -> dump(&this -> data.top_children, &ch, is_lepx); 

        // ---------- matching truth jets -------- //
        std::vector<particle_template*> tjets = tpx -> truthjets; 
        for (size_t y(0); y < tjets.size(); ++y){
            particle_template* ptr = tjets[y]; 
            std::map<std::string, particle_template*> prnt = ptr -> parents; 
            num_merged.push_back(int(prnt.size()));  
            num_jets += 1; 
        }

        if (tjets.size()){
            merge_data(&tjets, &nu);
            merge_data(&tjets, &lep); 
            this -> dump(&this -> data.top_truthjets, &tjets, is_lepx, &num_jets, &num_merged); 
        }

        // ---------- matching jets truth children -------- //
        num_jets = 0; 
        num_merged = {}; 

        std::vector<particle_template*> _jets = tpx -> jets;
        for (size_t y(0); y < _jets.size(); ++y){
            particle_template* ptr = _jets[y]; 
            std::map<std::string, particle_template*> prnt = ptr -> parents; 
            if (!prnt.count(tpx -> hash)){continue;}
            num_merged.push_back(int(prnt.size()));  
            num_jets += 1; 
        }

        if (_jets.size()){
            merge_data(&_jets, &nu); 
            merge_data(&_jets, &lep); 
            this -> dump(&this -> data.top_jets_children, &_jets, is_lepx, &num_jets, &num_merged); 
        }

        // ---------- matching jets leptons -------- //
        std::vector<particle_template*> jets_lepton = tpx -> jets;
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
            break;
        }
        if (!jets_lepton.size()){continue;}
        if (!is_lepx){
            this -> dump(&this -> data.top_jets_leptons, &jets_lepton, is_lepx, &num_jets, &num_merged); 
            continue;
        }
        if (jets_lepton.size() < 2){continue;}
        merge_data(&jets_lepton, &nu); 
        this -> dump(&this -> data.top_jets_leptons, &jets_lepton, is_lepx, &num_jets, &num_merged); 
    }
}
