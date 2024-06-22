#ifndef FEATURES_BSM_4TOPS_H
#define FEATURES_BSM_4TOPS_H

// ----------------- graph features ------------------ //
void static signal(bool* o, bsm_4tops* event){
    if (event -> Tops.size() != 4){*o = false; return;}
    std::vector<particle_template*> t = event -> Tops;
    for (size_t x(0); x < t.size(); ++x){
        top* tx = (top*)t[x]; 
        if (tx -> from_res){*o = true; return;}
    }
    *o = false; 
};

void static ntops(int* o, bsm_4tops* event){
    *o = event -> Tops.size(); 
    if (*o > 4){*o = 4;}
}; 

void static missingET(float* o, bsm_4tops* event){
    *o = event -> met; 
}; 

void static missingPhi(float* o, bsm_4tops* event){
    *o = event -> phi; 
}; 

void static n_nu(int* o, bsm_4tops* event){
    std::vector<particle_template*> ch = event -> Children; 
    int nus = 0; 
    for (size_t x(0); x < ch.size(); ++x){
        top_children* ch_ = (top_children*)ch[x]; 
        if (!(bool)ch_ -> is_nu){continue;}
        nus++; 
    }
    *o = nus; 
}; 

void static n_lep(int* o, bsm_4tops* event){
    std::vector<particle_template*> ch = event -> Children; 
    int lep = 0; 
    for (size_t x(0); x < ch.size(); ++x){
        top_children* ch_ = (top_children*)ch[x]; 
        if (!(bool)ch_ -> is_lep || (bool)ch_ -> is_nu){continue;}
        lep++; 
    }
    *o = lep; 
}; 







// -------------------- Node Features ---------------------- //
void static pt(double* o, particle_template* p_i){
    *o = p_i -> pt; 
}; 

void static eta(double* o, particle_template* p_i){
    *o = p_i -> eta; 
}; 

void static phi(double* o, particle_template* p_i){
    *o = p_i -> phi; 
}; 

void static energy(double* o, particle_template* p_i){
    *o = p_i -> e; 
}; 

void static is_lep(int* o, particle_template* p_i){
    *o = p_i -> is_lep; 
}; 

void static is_b(int* o, particle_template* p_i){
    *o = p_i -> is_b;
}; 

void static is_nu(int* o, particle_template* p_i){
    *o = p_i -> is_nu;
}; 

template <typename X>
void static res_node(int* o, X* p_i){
    *o = (int)p_i -> from_res; 
}; 

// ----------------- Edge Features ----------------------- //

template <typename X>
void static res_edge(int* o, std::tuple<X*, X*>* p_ij){
    X* pi = std::get<0>(*p_ij);
    X* pj = std::get<1>(*p_ij); 
    *o = ((int)pi -> from_res) * ((int)pj -> from_res); 
}; 

bool static res_edge(particle_template* p1, particle_template* p2){
    std::string n1 = p1 -> type;
    std::string n2 = p2 -> type; 
    bool r1, r2 = false;  

    if (n1 == "top"){r1 = ((top*)p1) -> from_res;}
    if (n2 == "top"){r2 = ((top*)p2) -> from_res;} 
    if (r1 && r2){return true;}

    if (n1 == "children"){r1 = ((top_children*)p1) -> from_res;}
    if (n2 == "children"){r2 = ((top_children*)p2)-> from_res;} 
    if (r1 && r2){return true;}

    if (n1 == "truthjets"){r1 = ((truthjet*)p1)-> from_res;}
    if (n2 == "truthjets"){r2 = ((truthjet*)p2)-> from_res;} 
    if (r1 && r2){return true;}
    return false; 
}; 





// ------------- top children edge truth ------------- //
void static top_edge(int* o, std::tuple<top_children*, top_children*>* p_ij){
    top_children* p1 = std::get<0>(*p_ij); 
    top_children* p2 = std::get<1>(*p_ij); 

    std::map<std::string, particle_template*> prt1 = p1 -> parents;
    std::map<std::string, particle_template*> prt2 = p2 -> parents; 
    
    if (!prt1.size() || !prt2.size()){*o = 0; return;}
    std::map<std::string, particle_template*>::iterator itr_1 = prt1.begin(); 
    *o = prt2.count(itr_1 -> first); 
}; 








#endif
