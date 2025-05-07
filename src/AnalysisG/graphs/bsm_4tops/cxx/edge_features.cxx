#include <bsm_4tops/edge_features.h>
#include <bsm_4tops/event.h>
#include <pyc/pyc.h>

void m_res_edge(int* o, top* t         ){*o *= t -> from_res;}
void m_res_edge(int* o, top_children* t){*o *= t -> from_res;}
void m_res_edge(int* o, truthjet* t    ){*o *= t -> from_res;}
void m_res_edge(int* o, jet* t         ){*o *= t -> from_res;}
void m_res_edge(int* o, electron* t    ){*o *= t -> from_res;}
void m_res_edge(int* o, muon* t        ){*o *= t -> from_res;}

void m_res_edge(int* o, neutrino* t, particle_template* ox){
    std::map<std::string, particle_template*> mx1 = t  -> parents; 
    if (mx1.count(std::string(ox -> hash))){return;}
    std::map<std::string, particle_template*> mx2 = ox -> parents; 
    std::map<std::string, particle_template*>::iterator itx = mx1.begin(); 
    for (; itx != mx1.end(); ++itx){
        if (!mx2.count(itx -> first)){continue;}
        return;
    }
    *o = 0; 
}

void res_edge(int* o, std::tuple<particle_template*, particle_template*>* pij){
    particle_template* p1 = std::get<0>(*pij); 
    particle_template* p2 = std::get<1>(*pij); 

    std::string type1 = p1 -> type;
    std::string type2 = p2 -> type; 

    *o = 1; 
    if (type1 == "top"){m_res_edge(o, (top*)p1);}
    if (type2 == "top"){m_res_edge(o, (top*)p2);}

    if (type1 == "children"){m_res_edge(o, (top_children*)p1);}
    if (type2 == "children"){m_res_edge(o, (top_children*)p2);}

    if (type1 == "truthjets"){m_res_edge(o, (truthjet*)p1);}
    if (type2 == "truthjets"){m_res_edge(o, (truthjet*)p2);}

    if (type1 == "jet"){m_res_edge(o, (jet*)p1);}
    if (type2 == "jet"){m_res_edge(o, (jet*)p2);}

    if (type1 == "el"){m_res_edge(o, (electron*)p1);}
    if (type2 == "el"){m_res_edge(o, (electron*)p2);}

    if (type1 == "mu"){m_res_edge(o, (muon*)p1);}
    if (type2 == "mu"){m_res_edge(o, (muon*)p2);}

    if (type1 == "nunu"){m_res_edge(o, (neutrino*)p1, p2);}
    if (type2 == "nunu"){m_res_edge(o, (neutrino*)p2, p1);}
}

int m_top_edge(top* t){return t -> index;}
int m_top_edge(top_children* t){return t -> top_index;}

std::vector<int> m_top_edge(jet* t){return t -> top_index;}
std::vector<int> m_top_edge(muon* t){return {t -> top_index};}
std::vector<int> m_top_edge(electron* t){return {t -> top_index};}
std::vector<int> m_top_edge(truthjet* t){return t -> top_index;}

void m_top_edge(neutrino* nx, particle_template* otr, int* o){
    std::map<std::string, particle_template*> mx1 = nx  -> parents; 
    if (mx1.count(std::string(otr -> hash))){*o = 1; return;}
    std::map<std::string, particle_template*> mx2 = otr -> parents; 
    std::map<std::string, particle_template*>::iterator itx = mx1.begin(); 
    for (; itx != mx1.end(); ++itx){
        if (!mx2.count(itx -> first)){continue;}
        *o = 1; return;
    }
    *o = 0; 
}

void top_edge(int* o, std::tuple<particle_template*, particle_template*>* pij){
    particle_template* p1 = std::get<0>(*pij); 
    particle_template* p2 = std::get<1>(*pij); 

    std::string type1 = p1 -> type;
    std::string type2 = p2 -> type; 

    std::vector<int> o1_ = {}; 
    if      (type1 == "top"      ){o1_.push_back(m_top_edge((top*)p1));}
    else if (type1 == "children" ){o1_.push_back(m_top_edge((top_children*)p1));}
    else if (type1 == "truthjets"){o1_ = m_top_edge((truthjet*)p1);}
    else if (type1 == "jet"      ){o1_ = m_top_edge((jet*)p1);}
    else if (type1 == "mu"       ){o1_ = m_top_edge((muon*)p1);}
    else if (type1 == "el"       ){o1_ = m_top_edge((electron*)p1);}
    else if (type1 == "nunu"     ){m_top_edge((neutrino*)p1, p2, o); return;}

    std::vector<int> o2_ = {};
    if      (type2 == "top"      ){o2_.push_back(m_top_edge((top*)p2));}
    else if (type2 == "children" ){o2_.push_back(m_top_edge((top_children*)p2));}
    else if (type2 == "truthjets"){o2_ = m_top_edge((truthjet*)p2);}
    else if (type2 == "jet"      ){o2_ = m_top_edge((jet*)p2);}
    else if (type2 == "mu"       ){o2_ = m_top_edge((muon*)p2);}
    else if (type2 == "el"       ){o2_ = m_top_edge((electron*)p2);}
    else if (type2 == "nunu"     ){m_top_edge((neutrino*)p2, p1, o); return;}

    *o = 0;  
    for (size_t x(0); x < o1_.size(); ++x){
        if (o1_[x] < 0){continue;}
        for (size_t y(0); y < o2_.size(); ++y){
            if (o2_[y] < 0){continue;}
            if (o1_[x] != o2_[y]){continue;}
            *o = 1; return; 
        }
    }
}


void det_res_edge(int* o, std::tuple<particle_template*, particle_template*>* pij){
    std::map<std::string, particle_template*> p1x = std::get<0>(*pij) -> parents; 
    std::map<std::string, particle_template*> p2x = std::get<1>(*pij) -> parents; 
    *o = 0; 

    std::map<std::string, particle_template*>::iterator itp; 
    for (itp = p1x.begin(); itp != p1x.end(); ++itp){
        if (!p2x.count(itp -> first)){continue;}
        *o = 1; return;
    }
}

void det_top_edge(int* o, std::tuple<particle_template*, particle_template*>* pij){
    std::map<std::string, particle_template*> p1x = std::get<0>(*pij) -> children; 
    std::map<std::string, particle_template*> p2x = std::get<1>(*pij) -> children; 

    *o = 0; 
    std::map<std::string, particle_template*>::iterator itp; 
    for (itp = p1x.begin(); itp != p1x.end(); ++itp){
        if (!p2x.count(itp -> first)){continue;}
        if (p1x.size() < 3 && p2x.size() < 3){return;}
        *o = 1; return;
    }
}



























