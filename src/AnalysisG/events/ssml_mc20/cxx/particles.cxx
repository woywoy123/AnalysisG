#include "particles.h"

zprime::zprime(){
    this -> type = "parton_Zprime"; 
    this -> add_leaf("mass", "_m"); 
    this -> add_leaf("eta" , "_eta"); 
    this -> add_leaf("phi" , "_phi");
    this -> add_leaf("pt"  , "_pt"); 
    this -> apply_type_prefix(); 
}

zprime::~zprime(){}

particle_template* zprime::clone(){return (particle_template*)new zprime();}

void zprime::build(std::map<std::string, particle_template*>* prt, element_t* el){
    float pt, eta, phi, en;  
    el -> get("pt"  , &pt); 
    el -> get("eta" , &eta); 
    el -> get("phi" , &phi); 
    el -> get("mass", &en); 

    zprime* prx = new zprime(); 
    prx -> pt   = pt; 
    prx -> eta  = eta; 
    prx -> phi  = phi; 
    prx -> mass = en; 
    (*prt)[prx -> hash] = prx; 
}


top::top(){
    this -> type = "parton_top"; 
    this -> add_leaf("mass", "_m"); 
    this -> add_leaf("eta" , "_eta"); 
    this -> add_leaf("phi" , "_phi");
    this -> add_leaf("pt"  , "_pt"); 
    this -> add_leaf("from_res"   , "_isFromZprime"); 
    this -> add_leaf("is_hadronic", "_isHadronic"); 
    this -> apply_type_prefix(); 
}

top::~top(){}

particle_template* top::clone(){return (particle_template*)new top();}

void top::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<top*> elc; 
    pmu_mass(&elc, el); 

    std::vector<int> from_res, is_had; 
    el -> get("from_res"   , &from_res); 
    el -> get("is_hadronic", &is_had);

    for (size_t x(0); x < elc.size(); ++x){
        top* elx = elc[x]; 
        elx -> index = int(x); 
        elx -> from_res = from_res[x]; 
        elx -> is_hadronic = is_had[x];
        
        (*prt)[elx -> hash] = elx; 
    }
}


parton_w1::parton_w1(){
    this -> type = "parton_Wdecay1"; 
    this -> add_leaf("mass" , "_m"); 
    this -> add_leaf("eta"  , "_eta"); 
    this -> add_leaf("phi"  , "_phi");
    this -> add_leaf("pt"   , "_pt"); 
    this -> add_leaf("pdgid", "_pdgid"); 
    this -> apply_type_prefix(); 
}

parton_w1::~parton_w1(){}

particle_template* parton_w1::clone(){return (particle_template*)new parton_w1();}

void parton_w1::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<parton_w1*> elc; 
    pmu_mass(&elc, el); 

    std::vector<int> pdgid; 
    el -> get("pdgid", &pdgid); 
    for (size_t x(0); x < elc.size(); ++x){
        parton_w1* elx = elc[x]; 
        elx -> index = x; 
        elx -> pdgid = pdgid[x]; 
        (*prt)[elx -> hash] = elx; 
    }
}

parton_w2::parton_w2(){
    this -> type = "parton_Wdecay2"; 
    this -> add_leaf("mass" , "_m"); 
    this -> add_leaf("eta"  , "_eta"); 
    this -> add_leaf("phi"  , "_phi");
    this -> add_leaf("pt"   , "_pt"); 
    this -> add_leaf("pdgid", "_pdgid"); 
    this -> apply_type_prefix(); 
}

parton_w2::~parton_w2(){}

particle_template* parton_w2::clone(){return (particle_template*)new parton_w2();}

void parton_w2::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<parton_w2*> elc; 
    pmu_mass(&elc, el); 

    std::vector<int> pdgid; 
    el -> get("pdgid", &pdgid); 
    for (size_t x(0); x < elc.size(); ++x){
        parton_w2* elx = elc[x]; 
        elx -> index = x; 
        elx -> pdgid = pdgid[x]; 
        (*prt)[elx -> hash] = elx; 
    }
}

parton_b::parton_b(){
    this -> type = "parton_b"; 
    this -> add_leaf("mass" , "_m"); 
    this -> add_leaf("eta"  , "_eta"); 
    this -> add_leaf("phi"  , "_phi");
    this -> add_leaf("pt"   , "_pt"); 
    this -> apply_type_prefix(); 
}

parton_b::~parton_b(){}

particle_template* parton_b::clone(){return (particle_template*)new parton_b();}

void parton_b::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<parton_b*> elc; 
    pmu_mass(&elc, el); 

    for (size_t x(0); x < elc.size(); ++x){
        parton_b* elx = elc[x]; 
        elx -> pdgid = 5; 
        elx -> index = x; 
        (*prt)[elx -> hash] = elx; 
    }
}

truthjet::truthjet(){
    this -> type = "truth_jet"; 
    this -> add_leaf("energy", "_e"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("pdgid" , "_partonid"); 
    this -> apply_type_prefix(); 
}

truthjet::~truthjet(){}

particle_template* truthjet::clone(){return (particle_template*)new truthjet();}

void truthjet::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<truthjet*> elc; 
    pmu(&elc, el); 
    std::vector<int> m_pdgid; 
    el -> get("pdgid", &m_pdgid); 
    for (size_t x(0); x < elc.size(); ++x){
        truthjet* elx = elc[x]; 
        if (m_pdgid.size()){elx -> pdgid = m_pdgid[x];}
        (*prt)[elx -> hash] = elx; 
    }
}

jet::jet(){
    this -> type = "jet"; 
    this -> add_leaf("energy", "_e_NOSYS"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt_NOSYS"); 

    this -> add_leaf("b_gnn_65" , "_GN2v01_FixedCutBEff_65_select"); 
    this -> add_leaf("b_gnn_70" , "_GN2v01_FixedCutBEff_70_select"); 
    this -> add_leaf("b_gnn_77" , "_GN2v01_FixedCutBEff_77_select"); 
    this -> add_leaf("b_gnn_85" , "_GN2v01_FixedCutBEff_85_select"); 
    this -> add_leaf("b_gnn_90" , "_GN2v01_FixedCutBEff_90_select"); 

    this -> add_leaf("top_index", "_truthTopIndex"); 
    this -> add_leaf("pdgid"    , "_partonid");
    this -> add_leaf("truthflav", "_truthflav");  

    this -> apply_type_prefix(); 

    this -> from_res.set_object(this); 
    this -> from_res.set_getter(this -> get_from_res);     
}

jet::~jet(){}

particle_template* jet::clone(){return (particle_template*)new jet();}

void jet::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<jet*> elc; 
    pmu(&elc, el); 

    std::vector<char> b_65, b_70, b_77, b_85, b_90; 
    el -> get("b_gnn_65", &b_65); 
    el -> get("b_gnn_70", &b_70); 
    el -> get("b_gnn_77", &b_77); 
    el -> get("b_gnn_85", &b_85); 
    el -> get("b_gnn_90", &b_90); 

    std::vector<int> tp_index, flav; 
    el -> get("top_index", &tp_index); 
    el -> get("truthflav", &flav);

    for (size_t x(0); x < elc.size(); ++x){
        jet* elx = elc[x]; 
        
        elx -> btag_65 = (bool)b_65[x]; 
        elx -> btag_70 = (bool)b_70[x]; 
        elx -> btag_77 = (bool)b_77[x]; 
        elx -> btag_85 = (bool)b_85[x]; 
        elx -> btag_90 = (bool)b_90[x]; 

        elx -> flav = flav[x]; 
         
        if (tp_index.size() && tp_index.size() > x){elx -> top_index = tp_index[x];}
        (*prt)[elx -> hash] = elx; 
    }
}

void jet::get_from_res(bool* val, jet* el){
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

muon::muon(){
    this -> type = "mu"; 
    this -> add_leaf("energy", "_e_NOSYS"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt_NOSYS"); 

    this -> add_leaf("charge"   , "_charge"); 
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

    for (size_t x(0); x < elc.size(); ++x){
        muon* elx = elc[x]; 
        elx -> charge = ch[x]; 
        elx -> is_lep = true; 
        if (tp_index.size() && tp_index.size() > x){elx -> top_index = tp_index[x];}
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
    this -> add_leaf("energy", "_e_NOSYS"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt_NOSYS"); 

    this -> add_leaf("charge"   , "_charge"); 
    this -> add_leaf("top_index", "_truthTopIndex"); 
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

    std::vector<int> tp_index; 
    el -> get("top_index", &tp_index); 

    for (size_t x(0); x < elc.size(); ++x){
        electron* elx = elc[x]; 
        elx -> charge = ch[x];
        elx -> is_lep = true;  
        if (tp_index.size() && tp_index.size() > x){elx -> top_index = tp_index[x];}
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





