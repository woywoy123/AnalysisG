#include <templates/particle_template.h>

void particle_template::set_pdgid(int* val, particle_template* prt){
    prt -> data.pdgid = *val; 
}

void particle_template::get_pdgid(int* val, particle_template* prt){
    particle_t* p = &prt -> data; 
    if (p -> pdgid != 0){ *val = p -> pdgid; return; }
    if (p -> symbol.size() == 0){ *val = p -> pdgid; return; }
    
    std::map<int, std::string> sym = {
             {1, "d"}, {2, "u"}, {3, "s"}, 
             {4, "c"}, {5, "b"}, {6, "t"},
             {11, "e"}, {12, "$\\nu_{e}$"}, 
             {13, "$\\mu$"}, {14, "$\\nu_{\\mu}$"}, 
             {15, "$\\tau$"}, {16, "$\\nu_{\\tau}$"},
             {21, "g"}, {22, "$\\gamma$"}
    }; 

    std::map<int, std::string>::iterator it = sym.begin(); 
    for (; it != sym.end(); ++it){
        if (it -> second != p -> symbol){continue;}
        p -> pdgid = it -> first; 
        *val = p -> pdgid; 
        return; 
    }
    *val = p -> pdgid;  
}

void particle_template::set_symbol(std::string* val, particle_template* prt){
    prt -> data.symbol = *val;
}

void particle_template::get_symbol(std::string* val, particle_template* prt){
    particle_t* p = &prt -> data; 
    if ((p -> symbol).size() != 0){*val = p -> symbol; return; }
    
    std::map<int, std::string> sym = {
             {1, "d"}, {2, "u"}, {3, "s"}, 
             {4, "c"}, {5, "b"}, {6, "t"},
             {11, "e"}, {12, "$\\nu_{e}$"}, 
             {13, "$\\mu$"}, {14, "$\\nu_{\\mu}$"}, 
             {15, "$\\tau$"}, {16, "$\\nu_{\\tau}$"},
             {21, "g"}, {22, "$\\gamma$"}
    }; 

    std::stringstream ss; 
    ss << sym[std::abs(p -> pdgid)];
    *val = ss.str(); 
}

void particle_template::set_charge(double* val, particle_template* prt){
    prt -> data.charge = *val;
}

void particle_template::get_charge(double* val, particle_template* prt){
    *val = prt -> data.charge;
}

void particle_template::get_isb(bool* val, particle_template* prt){ 
    *val = prt -> is({5}); 
}

void particle_template::get_isnu(bool* val, particle_template* prt){ 
    *val = prt -> is(prt -> data.nudef); 
}

void particle_template::get_islep(bool* val, particle_template* prt){ 
    *val = prt -> is(prt -> data.lepdef); 
}

void particle_template::get_isadd(bool* val, particle_template* prt){ 
    *val = !(prt -> is_lep || prt -> is_nu || prt -> is_b); 
}


bool particle_template::is(std::vector<int> p){
    for (int i : p){ 
        if (std::abs(i) != std::abs(this -> data.pdgid)){continue;} 
        return true;
    }
    return false; 
}

void particle_template::get_lepdecay(bool* val, particle_template* prt){
    bool nu  = false; 
    bool lep = false; 
    std::map<std::string, particle_template*> tmp = prt -> children; 
    std::map<std::string, particle_template*>::iterator itr = tmp.begin(); 
    for (; itr != tmp.end(); ++itr){
        if (!nu) { nu  = itr -> second -> is_nu;}
        if (!lep){ lep = itr -> second -> is_lep;}
    }
    if (lep && nu){ *val = true; return;}
    *val = false; 
}

void particle_template::add_leaf(std::string key, std::string leaf){
    this -> leaves[key] = leaf; 
}

void particle_template::set_parents(std::map<std::string, particle_template*>* val, particle_template* prt){
    std::map<std::string, particle_template*>::iterator itr = val -> begin();
    for (; itr != val -> end(); ++itr){prt -> register_parent(itr -> second);}
    if (val -> size()){return;}
    prt -> data.parents = {}; 
}

void particle_template::get_parents(std::map<std::string, particle_template*>* val, particle_template* prt){
    std::map<std::string, particle_template*> tmp = {}; 
    std::map<std::string, particle_template*>::iterator itr = val -> begin();
    for (; itr != val -> end(); ++itr){
        if (tmp.count(itr -> second -> hash)){continue;}
        tmp[itr -> second -> hash] = itr -> second; 
    }
    *val = tmp; 
}

void particle_template::set_children(std::map<std::string, particle_template*>* val, particle_template* prt){
    std::map<std::string, particle_template*>::iterator itr = val -> begin();
    for (; itr != val -> end(); ++itr){prt -> register_child(itr -> second);}
    if (val -> size()){return;}
    prt -> data.children = {}; 
}

void particle_template::get_children(std::map<std::string, particle_template*>* val, particle_template* prt){
    std::map<std::string, particle_template*> tmp = {}; 
    std::map<std::string, particle_template*>::iterator itr = val -> begin();
    for (; itr != val -> end(); ++itr){
        if (tmp.count(itr -> second -> hash)){continue;}
        tmp[itr -> second -> hash] = itr -> second; 
    }
    *val = tmp; 
}

bool particle_template::register_child(particle_template* p){
    std::string hash = p -> hash; 
    if (this -> data.children[hash]){return false;}
    this -> m_children[hash] = p; 
    this -> data.children[hash] = true; 
    return true; 
}

bool particle_template::register_parent(particle_template* p){
    std::string hash = p -> hash; 
    if (this -> data.parents[hash]){return true;}
    this -> m_parents[hash] = p; 
    this -> data.parents[hash] = true; 
    return true; 
}

void particle_template::get_hash(std::string* val, particle_template* prt){
    particle_t* p = &prt -> data; 
    if ((p -> hash).size()){*val = p -> hash; return; }

    prt -> to_cartesian(); 
    p -> hash  = prt -> to_string(prt -> px); 
    p -> hash += prt -> to_string(prt -> py); 
    p -> hash += prt -> to_string(prt -> pz);
    p -> hash += prt -> to_string(prt -> e); 
    p -> hash  = prt -> tools::hash(p -> hash, 18); 
    *val = p -> hash; 
}

void particle_template::set_type(std::string* val, particle_template* prt){
    prt -> data.type = *val;  
}

void particle_template::get_type(std::string* val, particle_template* prt){
    *val = prt -> data.type; 
}

void particle_template::set_index(int* val, particle_template* prt){
    prt -> data.index = *val;  
}

void particle_template::get_index(int* val, particle_template* prt){
    *val = prt -> data.index; 
}


