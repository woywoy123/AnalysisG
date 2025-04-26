#include "combinatorial.h"
#include <bsm_4tops/event.h>
#include <pyc/pyc.h>

packet_t::~packet_t(){
    for (size_t x(0); x < this -> nu1.size(); ++x){delete this -> nu1[x];}
    for (size_t x(0); x < this -> nu2.size(); ++x){delete this -> nu2[x];}
}


combinatorial::combinatorial(){this -> name = "combinatorial";}
combinatorial::~combinatorial(){this -> safe_delete(&this -> storage);}

selection_template* combinatorial::clone(){
    combinatorial* cl = new combinatorial(); 
    cl -> num_device = this -> num_device;
    return cl;
}

void combinatorial::merge(selection_template* sl){
    combinatorial* slt = (combinatorial*)sl; 

    for (size_t x(0); x < slt -> storage.size(); ++x){
        packet_t* tc = slt -> storage[x];
        std::vector<particle_template*> vout = {}; 
        merge_data(&vout, &tc -> t_bquarks); 
        merge_data(&vout, &tc -> t_leptons); 
        merge_data(&vout, &tc -> t_neutrino); 
        this -> write(&vout, tc -> name, particle_enum::pmu); 
        this -> write(&vout, tc -> name, particle_enum::pdgid); 
        this -> write(&tc -> matched_bquarks, tc -> name + "_matched_bquark"); 
        this -> write(&tc -> matched_leptons, tc -> name + "_matched_lepton"); 
        this -> write(&tc -> distance  , tc -> name + "_distance");
        this -> write(&tc -> chi2_nu1  , tc -> name + "_nu1_chi2"); 
        this -> write(&tc -> chi2_nu2  , tc -> name + "_nu2_chi2"); 
        this -> write(&tc -> nu1, tc -> name + "_nu1", particle_enum::pmu); 
        this -> write(&tc -> nu2, tc -> name + "_nu2", particle_enum::pmu); 
    }
}

bool combinatorial::selection(event_template* ev){
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
    return true; //num_leps == 2;  
}
   

double chi2(neutrino* nu, particle_template* nut){
    if (!nut){return -1;}
    double ch = std::pow(nu -> px - nut -> px, 2); 
    ch += std::pow(nu -> py - nut -> py, 2); 
    ch += std::pow(nu -> pz - nut -> pz, 2); 
    return ch; 
}


particle_template* safe_get(std::vector<particle_template*>* data, int idx){
    return (data -> size() != 2) ? nullptr : data -> at(idx); 
}

void combinatorial::reconstruction(packet_t* data){
    std::vector<std::vector<particle_template*>> particles = {this -> make_unique(&data -> particles)};
    if (!particles.size()){return;}
    std::vector<std::pair<neutrino*, neutrino*>> nux = pyc::nusol::combinatorial(
            std::vector<double>({data -> met}), std::vector<double>({data -> phi}), 
            particles, data -> device, this -> masstop, this -> massw, 
            data -> null, data -> perturb, data -> steps
    ); 

    if (!nux.size()){return;}
    particle_template* nu1_t = safe_get(&data -> t_neutrino, 0); 
    particle_template* nu2_t = safe_get(&data -> t_neutrino, 1); 

    particle_template* bqrk1 = safe_get(&data -> t_bquarks , 0); 
    particle_template* bqrk2 = safe_get(&data -> t_bquarks , 1); 

    particle_template* leps1 = safe_get(&data -> t_leptons , 0); 
    particle_template* leps2 = safe_get(&data -> t_leptons , 1); 


    for (size_t x(0); x < nux.size(); ++x){
        neutrino* nu1 = std::get<0>(nux[x]);
        neutrino* nu2 = std::get<1>(nux[x]); 

        data -> nu1.push_back(nu1); 
        data -> nu2.push_back(nu2); 
        data -> distance.push_back(nu1 -> min);  

        double nu1_chi2_e = chi2(nu1, nu1_t); 
        double nu1_chi2_s = chi2(nu2, nu1_t); 

        double nu2_chi2_e = chi2(nu1, nu2_t); 
        double nu2_chi2_s = chi2(nu2, nu2_t); 

        data -> chi2_nu1.push_back((nu1_chi2_e < nu1_chi2_s) ? nu1_chi2_e : nu1_chi2_s); 
        data -> chi2_nu2.push_back((nu2_chi2_e < nu2_chi2_s) ? nu2_chi2_e : nu2_chi2_s); 

        data -> matched_bquarks.push_back(0);
        data -> matched_leptons.push_back(0);

        if (!nu1_t || !bqrk1 || !leps1){continue;}

        bool found = true; 
        found *= std::string(nu1 -> bquark -> hash) == std::string(bqrk1 -> hash); 
        found *= std::string(nu2 -> bquark -> hash) == std::string(bqrk2 -> hash); 
        if (found){data -> matched_bquarks[x] = 1;}

        // ------ swapped case -------- //
        found = true; 
        found *= std::string(nu1 -> bquark -> hash) == std::string(bqrk2 -> hash);
        found *= std::string(nu2 -> bquark -> hash) == std::string(bqrk1 -> hash); 
        if (found){data -> matched_bquarks[x] = -1;}

        found = true; 
        found *= std::string(nu1 -> lepton -> hash) == std::string(leps1 -> hash); 
        found *= std::string(nu2 -> lepton -> hash) == std::string(leps2 -> hash); 
        if (found){data -> matched_leptons[x] = 1;}

        // ------ swapped case -------- //
        found = true; 
        found *= std::string(nu1 -> lepton -> hash) == std::string(leps2 -> hash);
        found *= std::string(nu2 -> lepton -> hash) == std::string(leps1 -> hash); 
        if (found){data -> matched_leptons[x] = -1;}
    }
}


void combinatorial::update_state(packet_t* data){
    if (!data){
        this -> b_qrk  = nullptr; 
        this -> lepton = nullptr; 
        this -> nu_tru = nullptr; 
    }

    if (!this -> nu_tru || !this -> b_qrk || !this -> lepton){return;}
    data -> t_bquarks.push_back(this -> b_qrk);  
    data -> t_leptons.push_back(this -> lepton);
    data -> t_neutrino.push_back(this -> nu_tru); 
}



bool combinatorial::strategy(event_template* ev){
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

    // ------------- Children ----------- //
    merge_data(&pkt_tc -> particles, &evn -> Children); 

    // ------------- Truth Jets + Children --------- //
    merge_data(&pkt_tj -> particles, &evn -> TruthJets);
    merge_data(&pkt_tj -> particles, &pkt_tc -> t_leptons); 
   
    // ------------- Jets + Children -------------- //
    merge_data(&pkt_jc -> particles, &evn -> Jets); 
    merge_data(&pkt_jc -> particles, &pkt_tc -> t_leptons); 

    // ------------- Jets + Leptons --------------- //
    merge_data(&pkt_jl -> particles, &evn -> DetectorObjects); 

    this -> reconstruction(pkt_tc);
    this -> reconstruction(pkt_tj);
    this -> reconstruction(pkt_jc);
    this -> reconstruction(pkt_jl);
    return true; 
}

