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





