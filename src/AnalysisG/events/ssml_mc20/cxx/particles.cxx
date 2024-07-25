#include "particles.h"

jet::jet(){
    this -> type = "jet"; 
    this -> add_leaf("gn2_btag_65", "_GN2v01_FixedCutBEff_65_select"); 
    this -> add_leaf("gn2_btag_70", "_GN2v01_FixedCutBEff_70_select"); 
    this -> add_leaf("gn2_btag_77", "_GN2v01_FixedCutBEff_77_select"); 
    this -> add_leaf("gn2_btag_85", "_GN2v01_FixedCutBEff_85_select"); 
    this -> add_leaf("gn2_btag_90", "_GN2v01_FixedCutBEff_90_select"); 
    this -> add_leaf("energy", "_e");
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("pt", "_pt_NOSYS"); 

    this -> apply_type_prefix(); 
}

jet::~jet(){}

particle_template* jet::clone(){return (particle_template*)new jet();}

void jet::build(std::map<std::string, particle_template*>* prt, element_t* el){
    
    std::vector<char> m_gn2_btag_65; 
    std::vector<char> m_gn2_btag_70; 
    std::vector<char> m_gn2_btag_77; 
    std::vector<char> m_gn2_btag_85; 
    std::vector<char> m_gn2_btag_90; 
    std::vector<float> m_energy;      
    std::vector<float> m_eta;         
    std::vector<float> m_phi;         
    std::vector<float> m_pt;          

    el -> get("gn2_btag_65", &m_gn2_btag_65); 
    el -> get("gn2_btag_70", &m_gn2_btag_70); 
    el -> get("gn2_btag_77", &m_gn2_btag_77); 
    el -> get("gn2_btag_85", &m_gn2_btag_85); 
    el -> get("gn2_btag_90", &m_gn2_btag_90); 
    el -> get("energy"     , &m_energy); 
    el -> get("eta"        , &m_eta); 
    el -> get("phi"        , &m_phi); 
    el -> get("pt"         , &m_pt); 

    for (int x(0); x < m_pt.size(); ++x){
        jet* p = new jet(); 
        p -> gn2_btag_65 = (bool)m_gn2_btag_65[x];
        p -> gn2_btag_70 = (bool)m_gn2_btag_70[x];
        p -> gn2_btag_77 = (bool)m_gn2_btag_77[x];
        p -> gn2_btag_85 = (bool)m_gn2_btag_85[x];
        p -> gn2_btag_90 = (bool)m_gn2_btag_90[x];
        p -> e      = m_energy[x];     
        p -> eta    = m_eta[x];        
        p -> phi    = m_phi[x];        
        p -> pt     = m_pt[x];         
        (*prt)[std::string(p -> hash)] = p; 
    }
}



lepton::lepton(){
    this -> type = "lepton"; 
    this -> add_leaf("energy", "_e");
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("pt", "_pt_NOSYS"); 
    this -> add_leaf("id", "_Id"); 
    this -> add_leaf("charge", "_charge"); 

    this -> apply_type_prefix(); 
}

lepton::~lepton(){}

particle_template* lepton::clone(){return (particle_template*)new lepton();}

void lepton::build(std::map<std::string, particle_template*>* prt, element_t* el){
   
    std::vector<int> m_id; 
    std::vector<float> m_charge; 
    std::vector<float> m_energy;      
    std::vector<float> m_eta;         
    std::vector<float> m_phi;         
    std::vector<float> m_pt;          

    el -> get("energy", &m_energy); 
    el -> get("eta"   , &m_eta); 
    el -> get("phi"   , &m_phi); 
    el -> get("pt"    , &m_pt); 
    el -> get("id"    , &m_id); 
    el -> get("charge", &m_charge); 

    for (int x(0); x < m_pt.size(); ++x){
        lepton* p   = new lepton(); 
        p -> e      = m_energy[x];     
        p -> eta    = m_eta[x];        
        p -> phi    = m_phi[x];        
        p -> pt     = m_pt[x];         
        p -> charge = m_charge[x]; 
        p -> pdgid  = m_id[x]; 
        (*prt)[std::string(p -> hash)] = p; 
    }
}

muon::muon(){
    this -> type = "mu"; 
    this -> add_leaf("energy", "_e");
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("pt", "_pt_NOSYS"); 
    this -> add_leaf("charge", "_charge"); 

    this -> apply_type_prefix(); 
}

muon::~muon(){}

particle_template* muon::clone(){return (particle_template*)new muon();}

void muon::build(std::map<std::string, particle_template*>* prt, element_t* el){
   
    std::vector<float> m_charge; 
    std::vector<float> m_energy;      
    std::vector<float> m_eta;         
    std::vector<float> m_phi;         
    std::vector<float> m_pt;          

    el -> get("energy", &m_energy); 
    el -> get("eta"   , &m_eta); 
    el -> get("phi"   , &m_phi); 
    el -> get("pt"    , &m_pt); 
    el -> get("charge", &m_charge); 

    for (int x(0); x < m_pt.size(); ++x){
        muon* p  = new muon(); 
        p -> e   = m_energy[x];     
        p -> eta = m_eta[x];        
        p -> phi = m_phi[x];        
        p -> pt  = m_pt[x];         
        p -> charge = m_charge[x]; 
        (*prt)[std::string(p -> hash)] = p; 
    }
}



electron::electron(){
    this -> type = "el"; 
    this -> add_leaf("energy", "_e");
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("pt", "_pt_NOSYS"); 
    this -> add_leaf("charge", "_charge"); 

    this -> apply_type_prefix(); 
}

electron::~electron(){}

particle_template* electron::clone(){return (particle_template*)new electron();}

void electron::build(std::map<std::string, particle_template*>* prt, element_t* el){
   
    std::vector<float> m_charge; 
    std::vector<float> m_energy;      
    std::vector<float> m_eta;         
    std::vector<float> m_phi;         
    std::vector<float> m_pt;          

    el -> get("energy", &m_energy); 
    el -> get("eta"   , &m_eta); 
    el -> get("phi"   , &m_phi); 
    el -> get("pt"    , &m_pt); 
    el -> get("charge", &m_charge); 

    for (int x(0); x < m_pt.size(); ++x){
        electron* p   = new electron(); 
        p -> e      = m_energy[x];     
        p -> eta    = m_eta[x];        
        p -> phi    = m_phi[x];        
        p -> pt     = m_pt[x];         
        p -> charge = m_charge[x]; 
        (*prt)[std::string(p -> hash)] = p; 
    }
}





















