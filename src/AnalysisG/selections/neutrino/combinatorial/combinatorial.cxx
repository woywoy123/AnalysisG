#include "combinatorial.h"

combinatorial::combinatorial(){this -> name = "combinatorial";}
combinatorial::~combinatorial(){}
selection_template* combinatorial::clone(){return (selection_template*)new combinatorial();}

void combinatorial::merge(selection_template* sl){
    combinatorial* slt = (combinatorial*)sl; 
    merge_data(&this -> output, &slt -> output);  
}

bool combinatorial::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev;
    std::vector<particle_template*> tops = evn -> Tops; 
    if (tops.size() != 4){return false;}
    int num_leps = 0; 
    for (size_t x(0); x < tops.size(); ++x){
        std::vector<particle_template*> ch_ = this -> vectorize(&tops[x] -> children);  
        for (size_t i(0); i < ch_.size(); ++i){
            bool lp = ch_[i] -> is_lep; 
            if (!lp){continue;}
            num_leps += lp;
            break; 
        }
    }
    return num_leps == 2; // || num_leps == 1;
}

std::vector<std::vector<neutrino*>> combinatorial::build_nus(
    std::vector<particle_template*>* bqs, std::vector<particle_template*>* leps, 
    double met, double phi
){
    std::vector<double> _phi = {phi}; 
    std::vector<double> _met = {met};
    std::vector<std::vector<particle_template*>> _bqs  = {*bqs}; 
    std::vector<std::vector<particle_template*>> _leps = {*leps}; 

    double mw = this -> massw; 
    double mt = this -> masstop;  

    std::vector<std::vector<neutrino*>> output; 
    std::vector<std::pair<neutrino*, neutrino*>> nus; 
    nus = pyc::nusol::combinatorial(_met, _phi, _bqs, _leps, "cuda:0", mt, mw, 1e-10, 100, this -> steps); 
    for (size_t x(0); x < nus.size(); ++x){output.push_back({std::get<0>(nus[x]), std::get<1>(nus[x])});}
    return output;
}

std::vector<std::vector<neutrino*>> combinatorial::get_baseline(
    std::vector<particle_template*>* bqs, std::vector<particle_template*>* lpt, 
    std::vector<double>* tps, std::vector<double>* wbs, double met, double phi
){
    std::vector<double> _phi = {phi}; 
    std::vector<double> _met = {met}; 

    std::vector<particle_template*> bquark1 = {bqs -> at(0)}; 
    std::vector<particle_template*> bquark2 = {bqs -> at(1)}; 
    std::vector<particle_template*> lepton1 = {lpt -> at(0)}; 
    std::vector<particle_template*> lepton2 = {lpt -> at(1)}; 
    std::vector<std::vector<double>> tm1 = {{(*tps)[0], (*wbs)[0]}}; 
    std::vector<std::vector<double>> tm2 = {{(*tps)[1], (*wbs)[1]}}; 

    std::vector<std::vector<neutrino*>> output; 
    std::vector<std::pair<neutrino*, neutrino*>> nus; 
    nus = pyc::nusol::NuNu(bquark1, bquark2, lepton1, lepton2, _met, _phi, tm1, tm2, "cuda:0", 1e-10, 1e-9, 1e-6, 1000);   
    for (size_t x(0); x < nus.size(); ++x){output.push_back({std::get<0>(nus[x]), std::get<1>(nus[x])});}
    return output;
}

bool combinatorial::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::string hash = evn -> hash; 

    // ------------ find the tops that decay leptonically --------------- //    
    std::vector<particle_template*> nus, leps, bs, tps; 
    std::vector<particle_template*> tops = evn -> Tops; 
    std::vector<double> tmass, wmass; 
    for (size_t x(0); x < tops.size(); ++x){
        particle_template* b_   = nullptr; 
        particle_template* nu_  = nullptr;
        particle_template* lep_ = nullptr; 

        std::vector<particle_template*> ch_ = this -> vectorize(&tops[x] -> children); 
        for (size_t i(0); i < ch_.size(); ++i){
            if (ch_[i] -> is_lep){lep_ = ch_[i]; continue;}
            if (ch_[i] -> is_nu){  nu_ = ch_[i]; continue;}
            if (ch_[i] -> is_b){    b_ = ch_[i]; continue;}
        }
        if (!b_ || !nu_ || !lep_){continue;}
        bs.push_back(b_);  
        nus.push_back(nu_); 
        leps.push_back(lep_); 

        particle_template* tpsx = nullptr; 
        this -> sum(&ch_, &tpsx); 
        tps.push_back(tpsx); 
        tmass.push_back(tops[x] -> mass); 
        wmass.push_back((*lep_ + *nu_).mass); 
        if (leps.size() == 2 && bs.size() == 2){break;}
    }

    particle_template* all_children = nullptr; 
    this -> sum(&evn -> Children, &all_children); 
    if (!all_children){return false;}

    particle_template* all_nus = nullptr; 
    this -> sum(&nus, &all_nus); 
    if (!all_nus){return false;}

    event_data* evx = &this -> output[hash]; 
    evx -> delta_met    = (all_children -> pt - evn -> met) / 1000; 
    evx -> delta_metnu  = (all_nus -> pt - evn -> met) / 1000; 
    evx -> observed_met = evn -> met / 1000; 
    evx -> neutrino_met = all_nus -> pt / 1000;

    for (size_t x(0); x < nus.size(); ++x){
        evx -> truth_neutrinos.push_back(new neutrino(nus[x]));
        evx -> bquark.push_back(new particle(bs[x]));
        evx -> lepton.push_back(new particle(leps[x])); 
        evx -> tops.push_back(new particle(tps[x])); 
    }

    evx -> cobs_neutrinos = this -> get_neutrinos(&bs, &leps,    evn -> met,     evn -> phi); 
    evx -> cmet_neutrinos = this -> get_neutrinos(&bs, &leps, all_nus -> pt, all_nus -> phi); 

    evx -> robs_neutrinos = this -> get_baseline(&bs, &leps, &tmass, &wmass,    evn -> met,     evn -> phi); 
    evx -> rmet_neutrinos = this -> get_baseline(&bs, &leps, &tmass, &wmass, all_nus -> pt, all_nus -> phi); 
    return true; 
}
