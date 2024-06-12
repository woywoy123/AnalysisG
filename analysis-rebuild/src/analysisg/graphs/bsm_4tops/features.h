#ifndef FEATURES_BSM_4TOPS_H
#define FEATURES_BSM_4TOPS_H

template <typename G>
bool static signal(G* event){
    if (event -> Tops.size() != 4){return false;}
    std::vector<particle_template*> t = event -> Tops;
    for (size_t x(0); x < t.size(); ++x){
        top* tx = (top*)t[x]; 
        if (tx -> from_res){return true;}
    }
    return false; 
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








#endif
