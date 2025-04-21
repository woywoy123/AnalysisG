#include <bsm_4tops/event.h>
#include "validation.h"
#include <pyc/pyc.h>

packet_t::~packet_t(){
    for (size_t x(0); x < this -> static_nu1.size(); ++x){delete this -> static_nu1[x];}
    for (size_t x(0); x < this -> static_nu2.size(); ++x){delete this -> static_nu2[x];}
    for (size_t x(0); x < this -> dynamic_nu1.size(); ++x){delete this -> dynamic_nu1[x];}
    for (size_t x(0); x < this -> dynamic_nu2.size(); ++x){delete this -> dynamic_nu2[x];}
}

validation::validation(){this -> name = "validation";}
selection_template* validation::clone(){
    validation* vl = new validation();
    vl -> num_device = this -> num_device;
    return vl;
}
validation::~validation(){this -> safe_delete(&this -> storage);}

void validation::merge(selection_template* sl){
    validation* slt = (validation*)sl; 

    for (size_t x(0); x < slt -> storage.size(); ++x){
        packet_t* tc = slt -> storage[x];
        this -> write(&tc -> objects, tc -> name, particle_enum::pmu );
        this -> write(&tc -> objects, tc -> name, particle_enum::pdgid); 
        this -> write(&tc -> static_nu1 , tc -> name + "_static_nu1" , particle_enum::pmu); 
        this -> write(&tc -> static_nu2 , tc -> name + "_static_nu2" , particle_enum::pmu); 
        this -> write(&tc -> dynamic_nu1, tc -> name + "_dynamic_nu1", particle_enum::pmu); 
        this -> write(&tc -> dynamic_nu2, tc -> name + "_dynamic_nu2", particle_enum::pmu); 
        this -> write(&tc -> static_distances , tc -> name + "_static_dst" ); 
        this -> write(&tc -> dynamic_distances, tc -> name + "_dynamic_dst"); 
        if (x){continue;}

        this -> write(&tc -> met, "met"); 
        this -> write(&tc -> phi, "phi"); 
    }
}

bool validation::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<top*> tops = this -> upcast<top>(&evn -> Tops);
    int num_leps = 0; 
    for (size_t x(0); x < tops.size(); ++x){
        std::map<std::string, particle_template*> ch = tops[x] -> children; 
        std::vector<top_children*> ch_ = this -> upcast<top_children>(&ch); 
        for (size_t i(0); i < ch_.size(); ++i){
            bool lp = ch_[i] -> is_lep;
            if (!lp){continue;}
            num_leps += lp; 
            break; 
        }
    }
    return num_leps == 2; //  || num_leps == 1;  
}


void validation::reconstruction(packet_t* data){
    if (data -> bquarks.size() != 2){return;}
    std::vector<double> mass_tops, mass_boson; 
    std::vector<std::pair<neutrino*, neutrino*>> nux; 

    for (size_t x(0); x <  2; ++x){
        std::vector<particle_template*> top; 
        top.push_back(data -> bquarks[x]);
        top.push_back(data -> leptons[x]); 
        top.push_back(data -> neutrino[x]); 

        particle_template* ptr_t = nullptr; 
        this -> sum(&top, &ptr_t);

        std::vector<particle_template*> wboson; 
        wboson.push_back(data -> leptons[x]); 
        wboson.push_back(data -> neutrino[x]); 

        particle_template* ptr_w = nullptr; 
        this -> sum(&wboson, &ptr_w); 

        mass_tops.push_back(ptr_t -> mass);
        mass_boson.push_back(ptr_w -> mass);
    }

    nux = pyc::nusol::NuNu(
            std::vector<particle_template*>({data -> bquarks[0]}), std::vector<particle_template*>({data -> bquarks[1]}),
            std::vector<particle_template*>({data -> leptons[0]}), std::vector<particle_template*>({data -> leptons[1]}),
            std::vector<double>({data -> met}), std::vector<double>({data -> phi}), 

            std::vector<std::vector<double>>({{mass_tops[0], mass_boson[0]}}), 
            std::vector<std::vector<double>>({{mass_tops[1], mass_boson[1]}}), 

            data -> device, data -> null, data -> step, data -> tolerance, data -> timeout
    ); 
    
    for (size_t x(0); x < nux.size(); ++x){
        data -> dynamic_nu1.push_back(std::get<0>(nux[x])); 
        data -> dynamic_nu2.push_back(std::get<1>(nux[x])); 
        data -> dynamic_distances.push_back(std::get<0>(nux[x]) -> min);  
    }

    nux = pyc::nusol::NuNu(
            std::vector<particle_template*>({data -> bquarks[0]}), std::vector<particle_template*>({data -> bquarks[1]}),
            std::vector<particle_template*>({data -> leptons[0]}), std::vector<particle_template*>({data -> leptons[1]}),
            std::vector<double>({data -> met}), std::vector<double>({data -> phi}), 

            std::vector<std::vector<double>>({{this -> masstop, this -> massw}}), 
            std::vector<std::vector<double>>({{this -> masstop, this -> massw}}), 

            data -> device, data -> null, data -> step, data -> tolerance, data -> timeout
    ); 
    
    for (size_t x(0); x < nux.size(); ++x){
        data -> static_nu1.push_back(std::get<0>(nux[x])); 
        data -> static_nu2.push_back(std::get<1>(nux[x])); 
        data -> static_distances.push_back(std::get<0>(nux[x]) -> min);  
    }
}

void validation::update_state(packet_t* data){
    if (!data){
        this -> b_qrk  = nullptr; 
        this -> lepton = nullptr; 
        this -> nu_tru = nullptr; 
    }

    if (!this -> nu_tru || !this -> b_qrk || !this -> lepton){return;}
    data -> bquarks.push_back(this -> b_qrk);  
    data -> leptons.push_back(this -> lepton);
    data -> neutrino.push_back(this -> nu_tru); 

    data -> objects.push_back(this -> b_qrk); 
    data -> objects.push_back(this -> lepton); 
    if (data -> name != "top_children"){return;}
    data -> objects.push_back(this -> nu_tru);
}

bool validation::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 

    std::vector<particle_template*> dleps = {}; 
    merge_data(&dleps, &evn -> Electrons); 
    merge_data(&dleps, &evn -> Muons); 

    packet_t* pkt_tc = this -> build_packet(evn, "top_children"); 
    packet_t* pkt_tj = this -> build_packet(evn, "truthjet"    ); 
    packet_t* pkt_jc = this -> build_packet(evn, "jetchildren" ); 
    packet_t* pkt_jl = this -> build_packet(evn, "jetleptons"  ); 

    std::vector<top*> tops = this -> upcast<top>(&evn -> Tops); 
    for (size_t x(0); x < tops.size(); ++x){
        this -> update_state(nullptr); // reset the global matching

        top* tpx = tops[x]; 
        std::vector<particle_template*> ch = this -> vectorize(&tpx -> children); 
        for (size_t c(0); c < ch.size(); ++c){
            if (ch[c] -> is_b){this -> b_qrk = ch[c];}
            if (ch[c] -> is_lep){this -> lepton = ch[c];}
            if (ch[c] -> is_nu){this -> nu_tru = ch[c];}
        }
        this -> update_state(pkt_tc); 
        this -> b_qrk  = nullptr;

        for (size_t c(0); c < tpx -> TruthJets.size(); ++c){
            truthjet* ptr = tpx -> TruthJets[c]; 
            if (!ptr -> is_b){continue;}
            this -> b_qrk = ptr; break; 
        }

        this -> update_state(pkt_tj); 
        this -> b_qrk  = nullptr;

        for (size_t c(0); c < tpx -> Jets.size(); ++c){
            jet* ptr = tpx -> Jets[c]; 
            if (!ptr -> is_b){continue;}
            this -> b_qrk = ptr; break; 
        }
        this -> update_state(pkt_jc); 
        this -> lepton = nullptr; 

        for (size_t c(0); c < dleps.size(); ++c){
            std::map<std::string, particle_template*> pr = dleps[c] -> parents; 
            for (size_t j(0); j < ch.size(); ++j){
                if (!ch[j] -> is_lep){continue;}
                if (!pr.count(ch[j] -> hash)){continue;}
                this -> lepton = dleps[c]; break; 
            }
            if (!this -> lepton){continue;}
            break;
        }
        this -> update_state(pkt_jl); 
    }

    this -> reconstruction(pkt_tc);
    this -> reconstruction(pkt_tj);
    this -> reconstruction(pkt_jc);
    this -> reconstruction(pkt_jl);
    return true; 
}

