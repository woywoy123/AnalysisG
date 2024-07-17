#include "decaymodes.h"

decaymodes::decaymodes(){
    this -> name = "decaymodes";
    this -> res_top_modes["HH"];
    this -> res_top_modes["HL"]; 
    this -> res_top_modes["LL"];

    this -> res_top_charges["SS"];
    this -> res_top_charges["SO"]; 

    this -> spec_top_modes["HH"];
    this -> spec_top_modes["HL"]; 
    this -> spec_top_modes["LL"];

    this -> spec_top_charges["SS"];
    this -> spec_top_charges["SO"]; 

    this -> signal_region["SS"]; 
    this -> signal_region["SO"]; 
}

decaymodes::~decaymodes(){}

selection_template* decaymodes::clone(){
    return (selection_template*)new decaymodes();
}

void decaymodes::merge(selection_template* sl){
    decaymodes* slt = (decaymodes*)sl; 

    merge_data(&this -> res_top_modes, &slt -> res_top_modes); 
    merge_data(&this -> res_top_charges, &slt -> res_top_charges); 
    merge_data(&this -> spec_top_modes, &slt -> spec_top_modes);
    merge_data(&this -> spec_top_charges, &slt -> spec_top_charges); 
    merge_data(&this -> ntops, &slt -> ntops); 
    merge_data(&this -> signal_region, &slt -> signal_region); 
    sum_data(&this -> res_top_pdgid, &slt -> res_top_pdgid); 
    sum_data(&this -> spec_top_pdgid, &slt -> spec_top_pdgid); 
    sum_data(&this -> all_pdgid, &slt -> all_pdgid); 
}

bool decaymodes::selection(event_template* ev){
    bsm_4tops* ev_ = (bsm_4tops*)ev; 
    std::vector<particle_template*> tops = ev_ -> Tops; 
    this -> ntops.push_back((int)tops.size());  
    
    int res = 0;
    int spc = 0;  
    for (size_t x(0); x < tops.size(); ++x){
        top* t = (top*)tops[x]; 
        res += t -> from_res; 
        spc += !(t -> from_res); 
    }
    return res == 2 && spc == 2;
}

bool decaymodes::strategy(event_template* ev){

    bsm_4tops* ev_ = (bsm_4tops*)ev; 
    std::vector<particle_template*> tops = ev_ -> Tops; 

    std::vector<particle_template*> rchildren = {}; 
    std::vector<particle_template*> rtops = {}; 

    std::vector<particle_template*> stops = {}; 
    std::vector<particle_template*> schildren = {}; 

    for (size_t x(0); x < tops.size(); ++x){
        top* t = (top*)tops[x]; 
        if (t -> from_res){rtops.push_back(tops[x]);}
        else {stops.push_back(tops[x]);}
    }

    std::map<std::string, int> mdicts = {}; 
    std::map<std::string, int> cdicts = {}; 

    // ------------ resonance tops ----------------- //
    for (size_t x(0); x < rtops.size(); ++x){
        top* t = (top*)rtops[x]; 
        bool ld = t -> lep_decay; 
        if (ld){mdicts["L"] += 1;}
        else {mdicts["H"] += 1;}

        std::map<std::string, particle_template*> c = t -> children; 
        std::vector<particle_template*> c_ = this -> vectorize(&c); 
        merge_data(&rchildren, &c_); 

        if (!ld){continue;}
        particle_template* lc_ = nullptr; 
        for (size_t l(0); l < c_.size(); ++l){
            if (!c_[l] -> is_lep || c_[l] -> is_nu){continue;}
            lc_ = c_[l]; 
            break;
        }
        if (!lc_){continue;}
        if (lc_ -> charge < 0){cdicts["S"] += 1;}
        else {cdicts["O"] += 1;}
    }
    
    std::string mheader = ""; 
    for (size_t x(0); x < mdicts["H"]; ++x){mheader += "H";}
    for (size_t x(0); x < mdicts["L"]; ++x){mheader += "L";}

    std::string cheader = ""; 
    if (cdicts["S"] == 2 || cdicts["O"] == 2){cheader = "SS";}
    if (cdicts["S"] == 1 && cdicts["O"] == 1){cheader = "SO";}

    double mass = this -> sum(&rchildren); 
    this -> res_top_modes[mheader].push_back(mass); 

    if (cheader.size()){this -> res_top_charges[cheader].push_back(mass);}

    for (size_t x(0); x < rchildren.size(); ++x){
        this -> res_top_pdgid[rchildren[x] -> symbol] += 1; 
        this -> all_pdgid[rchildren[x] -> symbol] += 1; 
    }


    // ------------ spectator tops -------------- //
    mdicts.clear(); 
    cdicts.clear(); 

    for (size_t x(0); x < stops.size(); ++x){
        top* t = (top*)stops[x]; 
        bool ld = t -> lep_decay; 
        if (ld){mdicts["L"] += 1;}
        else {mdicts["H"] += 1;}

        std::map<std::string, particle_template*> c = t -> children; 
        std::vector<particle_template*> c_ = this -> vectorize(&c); 
        merge_data(&schildren, &c_); 

        if (!ld){continue;}
        particle_template* lc_ = nullptr; 
        for (size_t l(0); l < c_.size(); ++l){
            if (!c_[l] -> is_lep || c_[l] -> is_nu){continue;}
            lc_ = c_[l]; 
            break;
        }
        if (!lc_){continue;}
        if (lc_ -> charge < 0){cdicts["S"] += 1;}
        else {cdicts["O"] += 1;}
    }
 
    mheader = ""; 
    for (size_t x(0); x < mdicts["H"]; ++x){mheader += "H";}
    for (size_t x(0); x < mdicts["L"]; ++x){mheader += "L";}

    cheader = ""; 
    if (cdicts["S"] == 2 || cdicts["O"] == 2){cheader = "SS";}
    if (cdicts["S"] == 1 && cdicts["O"] == 1){cheader = "SO";}

    mass = this -> sum(&schildren); 
    this -> spec_top_modes[mheader].push_back(mass); 
    if (cheader.size()){this -> spec_top_charges[cheader].push_back(mass);}

    for (size_t x(0); x < schildren.size(); ++x){
        this -> spec_top_pdgid[schildren[x] -> symbol] += 1; 
        this -> all_pdgid[schildren[x] -> symbol] += 1; 
    }

    std::map<std::string, int> signs = {}; 
    for (size_t x(0); x < rchildren.size(); ++x){
        if (!rchildren[x] -> is_lep || rchildren[x] -> is_nu){continue;}
        if (double(rchildren[x] -> charge) == 0){continue;}
        if (rchildren[x] -> charge < 0){signs["S"] += 1;}
        else {signs["O"] += 1;}
    }
    
    for (size_t x(0); x < schildren.size(); ++x){
        if (!schildren[x] -> is_lep || schildren[x] -> is_nu){continue;}
        if (double(schildren[x] -> charge) == 0){continue;}
        if (schildren[x] -> charge < 0){signs["S"] += 1;}
        else {signs["O"] += 1;}
    }
    
    cheader = ""; 
    if (signs["S"] == 2 || signs["O"] == 2){cheader = "SS";}
    if (signs["S"] == 1 && signs["O"] == 1){cheader = "SO";}
    if (!cheader.size()){return true;}
    this -> signal_region[cheader].push_back(this -> sum(&rchildren));
    return true; 
}

