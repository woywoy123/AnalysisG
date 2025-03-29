#include "particles.h"

jet::jet(){
    this -> type = "jet"; 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi");
    this -> add_leaf("pt"    , "_pt_NOSYS"); 
    this -> add_leaf("energy", "_e_NOSYS"); 
    this -> add_leaf("flav"  , "_truthflav");

    this -> add_leaf("b_gnn_77" , "_GN2v01_FixedCutBEff_77_select"); 
    this -> add_leaf("b_gnn_85" , "_GN2v01_FixedCutBEff_85_select"); 
    this -> add_leaf("b_gnn_90" , "_GN2v01_FixedCutBEff_90_select"); 
    this -> add_leaf("b_sel_85" , "_select_GN2v01_FixedCutBEff_85_NOSYS"); 
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

    std::vector<char> b_77, b_85, b_90, s_85; 
    el -> get("b_gnn_77", &b_77); 
    el -> get("b_gnn_85", &b_85); 
    el -> get("b_gnn_90", &b_90); 
    el -> get("b_sel_85", &s_85); 

    std::vector<int> tp_index, flav; 
    el -> get("top_index", &tp_index); 
    el -> get("flav"     , &flav    ); 

    bool sp = tp_index.size() == elc.size(); 
    for (size_t x(0); x < elc.size(); ++x){
        jet* elx = elc[x]; 
        elx -> btag_77 = (bool)b_77[x]; 
        elx -> btag_85 = (bool)b_85[x]; 
        elx -> btag_90 = (bool)b_90[x]; 
        elx -> sel_85  = (bool)s_85[x]; 
        elx -> pdgid   = flav[x]; 

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



