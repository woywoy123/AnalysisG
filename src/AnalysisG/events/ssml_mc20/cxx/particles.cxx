#include "particles.h"

zboson::zboson(){
    this -> type = "parton_Zprime"; 
    this -> add_leaf("mass", "_m"); 
    this -> add_leaf("eta" , "_eta"); 
    this -> add_leaf("phi" , "_phi");
    this -> add_leaf("pt"  , "_pt"); 
    this -> apply_type_prefix(); 
}

zboson::~zboson(){}

particle_template* zboson::clone(){return (particle_template*)new zboson();}

void zboson::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<float> pt, eta, phi, en;  
    el -> get("pt"  , &pt); 
    el -> get("eta" , &eta); 
    el -> get("phi" , &phi); 
    el -> get("mass", &en); 

    if (!pt.size()){return;}
    zboson* prx = new zboson(); 
    prx -> pt   = pt[0]; 
    prx -> eta  = eta[0]; 
    prx -> phi  = phi[0]; 
    prx -> mass = en[0]; 
    (*prt)[prx -> hash] = prx; 
}

top::top(){
    this -> type = "parton_top"; 
    this -> add_leaf("eta" , "_eta"); 
    this -> add_leaf("phi" , "_phi");
    this -> add_leaf("pt"  , "_pt"); 
    this -> add_leaf("mass", "_m"); 
    this -> add_leaf("from_res"   , "_isFromZprime"); 
    this -> apply_type_prefix(); 
}

top::~top(){}

particle_template* top::clone(){return (particle_template*)new top();}

void top::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<top*> elc; 
    pmu_mass(&elc, el); 

    std::vector<int> from_res; 
    el -> get("from_res", &from_res); 

    for (size_t x(0); x < elc.size(); ++x){
        top* elx = elc[x]; 
        elx -> index = int(x); 
        elx -> from_res = from_res[x]; 
        (*prt)[elx -> hash] = elx; 
    }
}


muon::muon(){
    this -> type = "mu"; 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("energy", "_e_NOSYS"); 
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
    bool sp = tp_index.size() == elc.size(); 
    for (size_t x(0); x < elc.size(); ++x){
        muon* elx = elc[x]; 
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
    bool sp = tp_index.size() == elc.size(); 
    for (size_t x(0); x < elc.size(); ++x){
        electron* elx = elc[x]; 
        elx -> charge = ch[x];
        elx -> is_lep = ch[x]*11;  
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

jet::jet(){
    this -> type = "jet"; 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt_NOSYS"); 
    this -> add_leaf("energy", "_e_NOSYS"); 

    this -> add_leaf("b_gnn_77" , "_GN2v01_FixedCutBEff_77_select"); 
    this -> add_leaf("b_gnn_85" , "_GN2v01_FixedCutBEff_85_select"); 
    this -> add_leaf("b_gnn_90" , "_GN2v01_FixedCutBEff_90_select"); 
    this -> add_leaf("top_index", "_truthTopIndex"); 
    this -> apply_type_prefix(); 

    this -> from_res.set_object(this); 
    this -> from_res.set_getter(this -> get_from_res);     
}

jet::~jet(){}

particle_template* jet::clone(){return (particle_template*)new jet();}

void jet::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<jet*> elc; 
    pmu(&elc, el); 

    std::vector<char> b_77, b_85, b_90; 
    el -> get("b_gnn_77", &b_77); 
    el -> get("b_gnn_85", &b_85); 
    el -> get("b_gnn_90", &b_90); 

    std::vector<int> tp_index; 
    el -> get("top_index", &tp_index); 

    bool sp = tp_index.size() == elc.size(); 
    for (size_t x(0); x < elc.size(); ++x){
        jet* elx = elc[x]; 
        elx -> btag_77 = (bool)b_77[x]; 
        elx -> btag_85 = (bool)b_85[x]; 
        elx -> btag_90 = (bool)b_90[x]; 
        if (sp){elx -> top_index = tp_index[x];}
        else {elx -> top_index = -2;}
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


truthjet::truthjet(){
    this -> type = "truth_jet"; 
    this -> add_leaf("energy", "_e"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt"); 
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



parton::parton(){
    this -> type = "parton"; 
    this -> add_leaf("w1_mass"         , "parton_Wdecay1_m"); 
    this -> add_leaf("w1_eta"          , "parton_Wdecay1_eta"); 
    this -> add_leaf("w1_phi"          , "parton_Wdecay1_phi");
    this -> add_leaf("w1_pt"           , "parton_Wdecay1_pt"); 
    this -> add_leaf("w1_pdgid"        , "parton_Wdecay1_pdgId"); 
    this -> add_leaf("wd1_el_index"    , "wd1_el_index"    ); 
    this -> add_leaf("wd1_mu_index"    , "wd1_mu_index"    ); 
    this -> add_leaf("wd1_recoj_index" , "wd1_recoj_index" ); 
    this -> add_leaf("wd1_truthj_index", "wd1_truthj_index"); 

    this -> add_leaf("w2_mass"         , "parton_Wdecay2_m"); 
    this -> add_leaf("w2_eta"          , "parton_Wdecay2_eta"); 
    this -> add_leaf("w2_phi"          , "parton_Wdecay2_phi");
    this -> add_leaf("w2_pt"           , "parton_Wdecay2_pt"); 
    this -> add_leaf("w2_pdgid"        , "parton_Wdecay2_pdgId"); 
    this -> add_leaf("wd2_el_index"    , "wd2_el_index"    ); 
    this -> add_leaf("wd2_mu_index"    , "wd2_mu_index"    ); 
    this -> add_leaf("wd2_recoj_index" , "wd2_recoj_index" ); 
    this -> add_leaf("wd2_truthj_index", "wd2_truthj_index"); 

    this -> add_leaf("b_mass"        , "parton_b_m"  ); 
    this -> add_leaf("b_eta"         , "parton_b_eta"); 
    this -> add_leaf("b_phi"         , "parton_b_phi");
    this -> add_leaf("b_pt"          , "parton_b_pt" ); 
    this -> add_leaf("b_recoj_index" , "b_recoj_index" ); 
    this -> add_leaf("b_truthj_index", "b_truthj_index"); 
}

parton::~parton(){}
particle_template* parton::clone(){return (particle_template*)new parton();}

void parton::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<int> pd, id_e, id_m, id_j, id_tj; 
    std::vector<float> mass, pt, eta, phi;   
    el -> get("w1_mass" , &mass); 
    el -> get("w1_eta"  , &eta); 
    el -> get("w1_phi"  , &phi); 
    el -> get("w1_pt"   , &pt); 
    el -> get("w1_pdgid", &pd);

    el -> get("wd1_el_index"    , &id_e); 
    el -> get("wd1_mu_index"    , &id_m);   
    el -> get("wd1_recoj_index" , &id_j); 
    el -> get("wd1_truthj_index", &id_tj);

    for (int x(0); x < pt.size(); ++x){
        parton* px = new parton(); 
        px -> index          = x; 
        px -> top_index      = x; 
        px -> pt             = pt[x]; 
        px -> eta            = eta[x]; 
        px -> phi            = phi[x]; 
        px -> mass           = mass[x]; 
        px -> pdgid          = pd[x]; 
        if (id_m.size()  == pt.size()){px -> muon_index     = id_m[x]; }
        if (id_e.size()  == pt.size()){px -> electron_index = id_e[x]; }
        if (id_j.size()  == pt.size()){px -> jet_index      = id_j[x]; }
        if (id_tj.size() == pt.size()){px -> truthjet_index = id_tj[x];} 
        (*prt)[px -> hash] = px; 
    }

    el -> get("w2_mass" , &mass); 
    el -> get("w2_eta"  , &eta); 
    el -> get("w2_phi"  , &phi); 
    el -> get("w2_pt"   , &pt); 
    el -> get("w2_pdgid", &pd);

    el -> get("wd2_el_index"    , &id_e); 
    el -> get("wd2_mu_index"    , &id_m);   
    el -> get("wd2_recoj_index" , &id_j); 
    el -> get("wd2_truthj_index", &id_tj);

    for (int x(0); x < pt.size(); ++x){
        parton* px = new parton(); 
        px -> index          = x; 
        px -> top_index      = x; 
        px -> pt             = pt[x]; 
        px -> eta            = eta[x]; 
        px -> phi            = phi[x]; 
        px -> mass           = mass[x]; 
        px -> pdgid          = pd[x]; 
        if (id_m.size()  == pt.size()){px -> muon_index     = id_m[x]; }
        if (id_e.size()  == pt.size()){px -> electron_index = id_e[x]; }
        if (id_j.size()  == pt.size()){px -> jet_index      = id_j[x]; }
        if (id_tj.size() == pt.size()){px -> truthjet_index = id_tj[x];} 
        (*prt)[px -> hash] = px; 
    }

    el -> get("b_mass"        , &mass); 
    el -> get("b_eta"         , &eta); 
    el -> get("b_phi"         , &phi); 
    el -> get("b_pt"          , &pt); 
    el -> get("b_recoj_index" , &id_j);
    el -> get("b_truthj_index", &id_tj); 

    for (int x(0); x < pt.size(); ++x){
        parton* px = new parton(); 
        px -> index          = x; 
        px -> top_index      = x; 
        px -> pt             = pt[x]; 
        px -> eta            = eta[x]; 
        px -> phi            = phi[x]; 
        px -> mass           = mass[x]; 
        px -> pdgid          = 5; 
        if (id_j.size()  == pt.size()){px -> jet_index      = id_j[x]; }
        if (id_tj.size() == pt.size()){px -> truthjet_index = id_tj[x];}         
        (*prt)[px -> hash] = px; 
    }

}

