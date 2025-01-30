#ifndef EDGE_FEATURES_SSML_MC20_H
#define EDGE_FEATURES_SSML_MC20_H

// --------------------- Edge Truth --------------------- //
void static m_res_edge(int* o, jet* t){*o *= t -> from_res;}
void static m_res_edge(int* o, electron* t){*o *= t -> from_res;}
void static m_res_edge(int* o, muon* t){*o *= t -> from_res;}
void static m_res_edge(int* o, particle_template* t){
    std::map<std::string, particle_template*> prt = t -> parents; 
    std::map<std::string, particle_template*>::iterator itr = prt.begin(); 

    int x = 0; 
    for (; itr != prt.end(); ++itr){
       top* t = (top*)itr -> second;  
       x += t -> from_res; 
       if (x){break;}
    }
    *o *= x; 
}

void static res_edge(int* o, std::tuple<particle_template*, particle_template*>* pij){
    particle_template* p1 = std::get<0>(*pij); 
    particle_template* p2 = std::get<1>(*pij); 

    std::string type1 = p1 -> type;
    std::string type2 = p2 -> type; 

    *o = 1; 

    if (type1 == "parton"){m_res_edge(o, p1);}
    if (type1 == "jet"){m_res_edge(o, (jet*)p1);}
    if (type1 == "el"){m_res_edge(o, (electron*)p1);}
    if (type1 == "mu"){m_res_edge(o, (muon*)p1);}

    if (type2 == "parton"){m_res_edge(o, p2);}
    if (type2 == "jet"){m_res_edge(o, (jet*)p2);}
    if (type2 == "el"){m_res_edge(o, (electron*)p2);}
    if (type2 == "mu"){m_res_edge(o, (muon*)p2);}
}

std::vector<int> static m_top_edge(jet* t){return {t -> top_index};}
std::vector<int> static m_top_edge(electron* t){return {t -> top_index};}
std::vector<int> static m_top_edge(muon* t){return {t -> top_index};}

void static top_edge(int* o, std::tuple<particle_template*, particle_template*>* pij){
    particle_template* p1 = std::get<0>(*pij); 
    particle_template* p2 = std::get<1>(*pij); 

    std::string type1 = p1 -> type;
    std::string type2 = p2 -> type; 

    std::vector<int> o1_ = {}; 
    std::vector<int> o2_ = {};
    if (type1 == "jet"){o1_ = m_top_edge((jet*)p1);}
    else if (type1 == "parton"){o1_ = {p1 -> index};}
    else if (type1 == "mu"){o1_ = m_top_edge((muon*)p1);}
    else if (type1 == "el"){o1_ = m_top_edge((electron*)p1);}

    if (type2 == "jet"){o2_ = m_top_edge((jet*)p2);}
    else if (type2 == "parton"){o1_ = {p2 -> index};}
    else if (type2 == "mu"){o2_ = m_top_edge((muon*)p2);}
    else if (type2 == "el"){o2_ = m_top_edge((electron*)p2);}
 
    *o = 0;  
    for (size_t x(0); x < o1_.size(); ++x){
        for (size_t y(0); y < o2_.size(); ++y){
            if (o1_[x] < 0 || o2_[x] < 0){continue;}
            if (o1_[x] != o2_[y]){continue;}
            *o = 1; return; 
        }
    }
}

#endif
