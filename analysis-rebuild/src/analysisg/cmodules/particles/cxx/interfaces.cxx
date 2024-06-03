#include <particles/particles.h>

void particles::symbol(std::string val){
    this -> data.symbol = val;
}

std::string particles::symbol(){
    particle_t* p = &this -> data; 
    if ((p -> symbol).size() != 0){return p -> symbol;}
    
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
    return ss.str(); 
}

void particles::pdgid(int val){
    this -> data.pdgid = val; 
}

int particles::pdgid(){
    particle_t* p = &this -> data; 
    if (p -> pdgid != 0){ return p -> pdgid; }
    if (p -> symbol.size() == 0){ return p -> pdgid; }
    
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
        return p -> pdgid; 
    }
    return p -> pdgid;  
}

void particles::charge(double val){
    this -> data.charge = val;
}

double particles::charge(){
    return this -> data.charge;
}

bool particles::is(std::vector<int> p){
    for (int i : p){ 
        if (std::abs(i) != std::abs(this -> data.pdgid)){continue;} 
        return true;
    }
    return false; 
}
bool particles::is_b(){ 
    return this -> is({5}); 
}

bool particles::is_nu(){ 
    return this -> is(this -> data.nudef); 
}

bool particles::is_lep(){ 
    return this -> is(this -> data.lepdef); 
}

bool particles::is_add(){ 
    bool out = (this -> is_lep() || this -> is_nu() || this -> is_b()); 
    return !out; 
}

bool particles::lep_decay(std::vector<particle_t>* inpt){
    bool nu  = false; 
    bool lep = false; 
    for (unsigned int x(0); x < inpt -> size(); ++x){
        particles* p = new particles(&inpt -> at(x)); 
        if (!nu) { nu  = p -> is_nu();}
        if (!lep){ lep = p -> is_lep();}
        delete p; 
    }
    if (lep && nu){ return true; }
    return false;
}

void particles::add_leaf(std::string key, std::string leaf){
    this -> leaves[key] = leaf; 
}

bool particles::register_child(particles* p){
    std::string hash = p -> hash(); 
    if (this -> data.children[hash]){return false;}
    this -> children[hash] = p; 
    this -> data.children[hash] = true; 
    return true; 
}

bool particles::register_parent(particles* p){
    std::string hash = p -> hash(); 
    if (this -> data.parents[hash]){return true;}
    this -> parents[hash] = p; 
    this -> data.parents[hash] = true; 
    return true; 
}

