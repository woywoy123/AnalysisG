#include "particles.h"

muon::muon(){
    this -> type = "mu"; 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("charge", "_charge"); 
    this -> add_leaf("pt"    , "_pt_NOSYS"); 
    this -> add_leaf("energy", "_e_NOSYS"); 
    this -> add_leaf("top_index", "_truthTopIndex"); 
    this -> apply_type_prefix(); 
   
    this -> from_res.set_object(this);  
    this -> from_res.set_getter(this -> get_from_res); 
}

muon::~muon(){}

particle_template* muon::clone(){return (particle_template*)new muon();}

void muon::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<muon*> elc; 
    pmu(&elc, el); 

    std::vector<float> ch; 
    el -> get("charge", &ch); 

    std::vector<int> tp_index; 
    el -> get("top_index", &tp_index); 
    bool sp = tp_index.size() == elc.size(); 
    for (size_t x(0); x < elc.size(); ++x){
        muon* elx = elc[x]; 
        elx -> index = x; 
        elx -> charge = ch[x]; 
        elx -> pdgid = ch[x]*(13); 
        if (sp){elx -> top_index = tp_index[x];}
        else {elx -> top_index = -2;}
        (*prt)[elx -> hash] = elx; 
    }
}

void muon::get_from_res(bool* val, muon* el){
    std::map<std::string, particle_template*> prnt = el -> parents; 
    std::map<std::string, particle_template*>::iterator itr = prnt.begin(); 
    for (; itr != prnt.end(); ++itr){
        top* t = (top*)itr -> second;
        if (!t -> from_res){continue;}
        *val = true;
        return; 
    }
    *val = false; 
}

electron::electron(){
    this -> type = "el"; 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt_NOSYS"); 
    this -> add_leaf("energy", "_e_NOSYS"); 

    this -> add_leaf("charge"   , "_charge"); 
    this -> add_leaf("top_index", "_truthTopIndex"); 
    this -> add_leaf("ecids"    , "_passECIDS_NOSYS"); 
    this -> apply_type_prefix(); 

    this -> from_res.set_object(this); 
    this -> from_res.set_getter(this -> get_from_res); 
}
electron::~electron(){}
particle_template* electron::clone(){return (particle_template*)new electron();}

void electron::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<electron*> elc; 
    pmu(&elc, el); 

    std::vector<float> ch; 
    el -> get("charge", &ch); 

    std::vector<int> tp_index, ecid; 
    el -> get("top_index", &tp_index); 
    el -> get("ecids"    , &ecid); 

    bool sp = tp_index.size() == elc.size(); 
    for (size_t x(0); x < elc.size(); ++x){
        electron* elx = elc[x]; 
        elx -> index  = x; 
        elx -> charge = ch[x];
        elx -> pdgid  = ch[x]*11;
        elx -> pass_ecids = ecid[x]; 

        if (sp){elx -> top_index = tp_index[x];}
        else {elx -> top_index = -2;}
        (*prt)[elx -> hash] = elx; 
    }
}

void electron::get_from_res(bool* val, electron* el){
    std::map<std::string, particle_template*> prnt = el -> parents; 
    std::map<std::string, particle_template*>::iterator itr = prnt.begin(); 
    for (; itr != prnt.end(); ++itr){
        top* t = (top*)itr -> second;
        if (!t -> from_res){continue;}
        *val = true;
        return; 
    }
    *val = false; 
}

lepton::lepton(){
    this -> type = "lepton"; 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt_NOSYS"); 
    this -> add_leaf("e"     , "_e_NOSYS"); 

    this -> add_leaf("charge", "_charge"); 
    this -> add_leaf("ambiguity", "_DFCommonAddAmbiguity"); 
    this -> apply_type_prefix(); 
}

lepton::~lepton(){}
particle_template* lepton::clone(){return (particle_template*)new lepton();}
void lepton::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<float> e, pt, eta, phi, ch;   
    el -> get("eta", &eta); 
    el -> get("phi", &phi); 
    el -> get("pt" , &pt); 
    el -> get("e"  , &e); 
    el -> get("charge", &ch);

    std::vector<int> df; 
    el -> get("ambiguity", &df);

    for (int x(0); x < pt.size(); ++x){
        lepton* px      = new lepton(); 
        px -> index     = x; 
        px -> pt        = pt[x]; 
        px -> eta       = eta[x]; 
        px -> phi       = phi[x]; 
        px -> e         = e[x]; 
        px -> charge    = ch[x]; 
        px -> ambiguity = df[x]; 
        (*prt)[px -> hash] = px; 
    }
}



