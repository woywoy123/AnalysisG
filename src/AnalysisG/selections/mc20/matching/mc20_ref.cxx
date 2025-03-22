#include "matching.h"
#include <ssml_mc20/event.h>

void matching::current(event_template* ev){
    ssml_mc20* evn = (ssml_mc20*)ev; 
    std::vector<particle_template*> dleps = evn -> Leptons; 
    std::vector<particle_template*> tops = evn -> Tops; 

    for (size_t x(0); x < tops.size(); ++x){
        top* tpx = (top*)tops[x]; 

        this -> data.top_partons.num_tops += 1; 
        this -> data.top_partons.mass.push_back(tpx -> mass); 

        std::map<std::string, particle_template*> chx = tpx -> children; 
        std::vector<particle_template*> ch = this -> vectorize(&chx); 
        if (!ch.size()){continue;}
      
        // ----------- matching children ---------- // 
        std::vector<particle_template*> nu = {}, lep = {}; 
        for (size_t y(0); y < ch.size(); ++y){
            if (ch[y] -> is_nu  &&  nu.size() == 0){ nu.push_back(ch[y]);}
            if (ch[y] -> is_lep && lep.size() == 0){lep.push_back(ch[y]);}
        }

        bool is_lepx = nu.size() > 0; 
        this -> data.top_partons.num_ltop +=  is_lepx;
        this -> data.top_partons.num_htop += !is_lepx;

        this -> data.top_children.pdgid.push_back(this -> get_pdgid(&ch));
        this -> data.top_children.is_leptonic.push_back(int( is_lepx)); 
        this -> data.top_children.is_hadronic.push_back(int(!is_lepx)); 
        this -> data.top_children.mass.push_back(this -> sum(&ch));

        this -> data.top_children.num_tops += 1; 
        this -> data.top_children.num_ltop +=  is_lepx;
        this -> data.top_children.num_htop += !is_lepx;


        // ---------- matching truth jets -------- //
        int num_jets = 0; 
        std::vector<int> num_merged = {}; 
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
            
            this -> data.top_truthjets.num_tops += 1; 
            this -> data.top_truthjets.num_ltop += is_lepx; 
            this -> data.top_truthjets.num_htop += !is_lepx; 

            this -> data.top_truthjets.is_leptonic.push_back(int( is_lepx)); 
            this -> data.top_truthjets.is_hadronic.push_back(int(!is_lepx)); 

            this -> data.top_truthjets.num_jets.push_back(num_jets); 
            this -> data.top_truthjets.merged.push_back(num_merged); 

            this -> data.top_truthjets.pdgid.push_back(this -> get_pdgid(&tjets)); 
            this -> data.top_truthjets.mass.push_back(this -> sum(&tjets)); 
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
            
            this -> data.top_jets_children.num_tops += 1; 
            this -> data.top_jets_children.num_ltop += is_lepx; 
            this -> data.top_jets_children.num_htop += !is_lepx; 

            this -> data.top_jets_children.is_leptonic.push_back(int( is_lepx)); 
            this -> data.top_jets_children.is_hadronic.push_back(int(!is_lepx)); 

            this -> data.top_jets_children.num_jets.push_back(num_jets); 
            this -> data.top_jets_children.merged.push_back(num_merged); 

            this -> data.top_jets_children.pdgid.push_back(this -> get_pdgid(&_jets)); 
            this -> data.top_jets_children.mass.push_back(this -> sum(&_jets)); 
        }

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

        if (jets_lepton.size()){
            merge_data(&jets_lepton, &nu); 

            this -> data.top_jets_leptons.num_tops += 1; 
            this -> data.top_jets_leptons.num_ltop += is_lepx; 
            this -> data.top_jets_leptons.num_htop += !is_lepx; 

            this -> data.top_jets_leptons.is_leptonic.push_back(int( is_lepx)); 
            this -> data.top_jets_leptons.is_hadronic.push_back(int(!is_lepx)); 

            this -> data.top_jets_leptons.num_jets.push_back(num_jets); 
            this -> data.top_jets_leptons.merged.push_back(num_merged); 

            this -> data.top_jets_leptons.pdgid.push_back(this -> get_pdgid(&jets_lepton)); 
            this -> data.top_jets_leptons.mass.push_back(this -> sum(&jets_lepton)); 
        }
    }
}


