#include "particles.h"

// ============================= Top ========================= //
top::top(){
    this -> type = "top"; 
    this -> add_leaf("pt" , "_pt"); 
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("e"  , "_e"); 
    this -> add_leaf("index", "_index");
    this -> add_leaf("pdgid", "_pdgid");
    this -> add_leaf("from_res", "_FromRes");
    this -> add_leaf("status", "_status");
    this -> apply_type_prefix(); 
}

particle_template* top::clone(){return (particle_template*)new top();}

void top::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<top*> out; 
    assign_vector(&out, el); 

    std::vector<int> _from_res, _status;
    el -> get("from_res", &_from_res); 
    el -> get("status"  , &_status); 
    
    for (int x(0); x < out.size(); ++x){
        top* t = out[x]; 
        t -> from_res = _from_res[x]; 
        t -> status = _status[x]; 
        (*prt)[std::string(t -> hash)] = t;
    } 
}

top::~top(){}

// ============================= Children ========================= //
top_children::top_children(){
    this -> type = "children"; 
    this -> add_leaf("pt" , "_pt"); 
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("e"  , "_e"); 
    this -> add_leaf("index", "_index");
    this -> add_leaf("pdgid", "_pdgid");

    this -> add_leaf("top_index", "_TopIndex");
    this -> apply_type_prefix(); 

    this -> from_res.set_object(this); 
    this -> from_res.set_getter(this -> get_from_res); 
}

void top_children::get_from_res(bool* val, top_children* prt){
    std::map<std::string, particle_template*> x_ = prt -> parents; 
    std::map<std::string, particle_template*>::iterator itr = x_.begin(); 
    for (; itr != x_.end(); ++itr){
        std::string tx = itr -> second -> type;
        if (tx != "top"){continue;}
        top* tp = (top*)itr -> second; 
        if (!tp -> from_res){continue;}
        *val = true;
        return; 
    }
    *val = false; 
}

particle_template* top_children::clone(){return (particle_template*)new top_children();}

void top_children::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<std::vector<int>> _index, _pdgid, _top_index; 
    std::vector<std::vector<float>> _pt, _eta, _phi, _e; 

    el -> get("top_index", &_top_index); 
    el -> get("index"    , &_index); 
    el -> get("pdgid"    , &_pdgid);
    
    el -> get("pt" , &_pt); 
    el -> get("eta", &_eta);
    el -> get("phi", &_phi); 
    el -> get("e"  , &_e); 

    for (int x(0); x < _pt.size(); ++x){
        for (int y(0); y < _pt[x].size(); ++y){
            top_children* p    = new top_children();
            p -> pt        = _pt[x][y]; 
            p -> eta       = _eta[x][y]; 
            p -> phi       = _phi[x][y]; 
            p -> e         = _e[x][y]; 
            p -> index     = _index[x][y]; 
            p -> pdgid     = _pdgid[x][y]; 
            p -> top_index = _top_index[x][y]; 
            (*prt)[std::string(p -> hash)] = p; 
        }
    }
}

top_children::~top_children(){}

// ============================= TruthJets ========================= //
truthjet::truthjet(){
    this -> type = "truthjets"; 
    this -> add_leaf("pt" , "_pt"); 
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("e"  , "_e"); 
    this -> add_leaf("index", "_index");
    this -> add_leaf("pdgid", "_btagged");
    this -> add_leaf("top_index", "_TopIndex");
    this -> add_leaf("top_quark_count", "_topquarkcount"); 
    this -> add_leaf("w_boson_count", "_wbosoncount"); 
    this -> apply_type_prefix(); 

    this -> from_res.set_object(this); 
    this -> from_res.set_getter(this -> get_from_res); 
}

particle_template* truthjet::clone(){return (particle_template*)new truthjet();}

void truthjet::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<truthjet*> out; 
    assign_vector(&out, el); 

    std::vector<std::vector<int>> _top_index; 
    std::vector<int> _top_count, _w_boson;

    el -> get("top_quark_count", &_top_count); 
    el -> get("w_boson_count"  , &_w_boson); 
    el -> get("top_index", &_top_index);  

    for (int x(0); x < _top_count.size(); ++x){
        truthjet* t          = out[x]; 
        t -> top_quark_count = _top_count[x]; 
        t -> w_boson_count   = _w_boson[x]; 
        t -> top_index       = _top_index[x]; 
        (*prt)[std::string(t -> hash)] = t;
    }
}

void truthjet::get_from_res(bool* val, truthjet* prt){
    if (!prt -> Tops.size()){*val = false; return;}

    *val = false; 
    for (int x(0); x < prt -> Tops.size(); ++x){
        if (!prt -> Tops[x] -> from_res){continue;} 
        *val = true;
        return; 
    }
}

truthjet::~truthjet(){}

// ============================= TruthJetParton ========================= //
truthjetparton::truthjetparton(){
    this -> type = "TJparton"; 
    this -> add_leaf("pt" , "_pt"); 
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("e"  , "_e"); 
    this -> add_leaf("index", "_index");
    this -> add_leaf("pdgid", "_pdgid");

    this -> add_leaf("charge", "_charge");
    this -> add_leaf("truthjet_index", "_TruthJetIndex");
    this -> add_leaf("topchild_index", "_ChildIndex");
    this -> apply_type_prefix(); 
}

particle_template* truthjetparton::clone(){return (particle_template*)new truthjetparton();}

void truthjetparton::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<truthjetparton*> out; 
    std::vector<float> _pt, _eta, _phi, _e, _pdgid, _charge; 
    el -> get("pt"    , &_pt); 
    el -> get("eta"   , &_eta);
    el -> get("phi"   , &_phi); 
    el -> get("e"     , &_e); 
    el -> get("pdgid" , &_pdgid);
    el -> get("charge", &_charge); 

    std::vector<int> _index, _truthjet_index; 
    el -> get("index"           , &_index); 
    el -> get("truthjet_index"  , &_truthjet_index); 

    std::vector<std::vector<int>> _topchild_index; 
    el -> get("topchild_index", &_topchild_index);  

    for (int x(0); x < _pt.size(); ++x){
        truthjetparton* p    = new truthjetparton();
        p -> pt              = _pt[x]; 
        p -> eta             = _eta[x]; 
        p -> phi             = _phi[x]; 
        p -> e               = _e[x]; 
        p -> charge          = _charge[x]; 
        p -> pdgid           = _pdgid[x]; 
    
        p -> index           = _index[x]; 
        p -> truthjet_index  = _truthjet_index[x]; 
        p -> topchild_index  = _topchild_index[x]; 
        (*prt)[std::string(p -> hash)] = p;
    }
}

truthjetparton::~truthjetparton(){}

// ============================= Jets ========================= //

jet::jet(){
    this -> type = "jet"; 
    this -> add_leaf("pt" , "_pt"); 
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("e"  , "_e"); 
    this -> add_leaf("index", "_index");

    this -> add_leaf("top_index"   , "_TopIndex"); 
    this -> add_leaf("btag_DL1r_60", "_isbtagged_DL1r_60"); 
    this -> add_leaf("btag_DL1_60" , "_isbtagged_DL1_60"); 

    this -> add_leaf("btag_DL1r_70", "_isbtagged_DL1r_70"); 
    this -> add_leaf("btag_DL1_70" , "_isbtagged_DL1_70"); 

    this -> add_leaf("btag_DL1r_77", "_isbtagged_DL1r_77"); 
    this -> add_leaf("btag_DL1_77" , "_isbtagged_DL1_77"); 

    this -> add_leaf("btag_DL1r_85", "_isbtagged_DL1r_85"); 
    this -> add_leaf("btag_DL1_85" , "_isbtagged_DL1_85"); 

    this -> add_leaf("DL1_b" , "_DL1_pb"); 
    this -> add_leaf("DL1_c" , "_DL1_pc"); 
    this -> add_leaf("DL1_u" , "_DL1_pu"); 

    this -> add_leaf("DL1r_b", "_DL1r_pb"); 
    this -> add_leaf("DL1r_c", "_DL1r_pc"); 
    this -> add_leaf("DL1r_u", "_DL1r_pu"); 

    this -> apply_type_prefix(); 
}

particle_template* jet::clone(){return (particle_template*)new jet();}

void jet::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<int> _index;  
    el -> get("index"     , &_index); 

    std::vector<std::vector<int>> _top_index; 
    el -> get("top_index" , &_top_index);
    
    std::vector<float> _pt, _eta, _phi, _e; 
    el -> get("pt" , &_pt); 
    el -> get("eta", &_eta);
    el -> get("phi", &_phi); 
    el -> get("e"  , &_e); 

    std::vector<char> _btag_DL1r_60; 
    std::vector<char> _btag_DL1_60;  
    std::vector<char> _btag_DL1r_70; 
    std::vector<char> _btag_DL1_70;  
    std::vector<char> _btag_DL1r_77; 
    std::vector<char> _btag_DL1_77;  
    std::vector<char> _btag_DL1r_85; 
    std::vector<char> _btag_DL1_85;  

    el -> get("btag_DL1r_60", &_btag_DL1r_60); 
    el -> get("btag_DL1_60" , &_btag_DL1_60);  
    el -> get("btag_DL1r_70", &_btag_DL1r_70); 
    el -> get("btag_DL1_70" , &_btag_DL1_70);  
    el -> get("btag_DL1r_77", &_btag_DL1r_77); 
    el -> get("btag_DL1_77" , &_btag_DL1_77);  
    el -> get("btag_DL1r_85", &_btag_DL1r_85); 
    el -> get("btag_DL1_85" , &_btag_DL1_85);  

    std::vector<float> _DL1_b; 
    std::vector<float> _DL1_c;  
    std::vector<float> _DL1_u; 
    std::vector<float> _DL1r_b; 
    std::vector<float> _DL1r_c; 
    std::vector<float> _DL1r_u;
    el -> get("DL1_b" , &_DL1_b ); 
    el -> get("DL1_c" , &_DL1_c ); 
    el -> get("DL1_u" , &_DL1_u ); 
    el -> get("DL1r_b", &_DL1r_b); 
    el -> get("DL1r_c", &_DL1r_c); 
    el -> get("DL1r_u", &_DL1r_u); 

    for (int x(0); x < _pt.size(); ++x){
        jet* p         = new jet();
        p -> pt        = _pt[x]; 
        p -> eta       = _eta[x]; 
        p -> phi       = _phi[x]; 
        p -> e         = _e[x]; 
        p -> index     = _index[x]; 
        p -> top_index = _top_index[x]; 

        p -> btag_DL1r_60 = (bool)_btag_DL1r_60[x];
        p -> btag_DL1_60  = (bool)_btag_DL1_60[x];
        p -> btag_DL1r_70 = (bool)_btag_DL1r_70[x];
        p -> btag_DL1_70  = (bool)_btag_DL1_70[x];
        p -> btag_DL1r_77 = (bool)_btag_DL1r_77[x];
        p -> btag_DL1_77  = (bool)_btag_DL1_77[x];
        p -> btag_DL1r_85 = (bool)_btag_DL1r_85[x];
        p -> btag_DL1_85  = (bool)_btag_DL1_85[x];

        p -> DL1_b  = _DL1_b[x]; 
        p -> DL1_c  = _DL1_c[x]; 
        p -> DL1_u  = _DL1_u[x]; 
        p -> DL1r_b = _DL1r_b[x]; 
        p -> DL1r_c = _DL1r_c[x]; 
        p -> DL1r_u = _DL1r_u[x]; 

        (*prt)[std::string(p -> hash)] = p; 
    }
}

jet::~jet(){}

// ============================= JetPartons ========================= //

jetparton::jetparton(){
    this -> type = "Jparton"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("e"     , "_e"); 
    this -> add_leaf("index" , "_index");
    this -> add_leaf("charge", "_charge"); 
    this -> add_leaf("pdgid" , "_pdgid"); 

    this -> add_leaf("JetIndex"     , "_JetIndex"); 
    this -> add_leaf("TopChildIndex", "_ChildIndex"); 


    this -> apply_type_prefix(); 
}

particle_template* jetparton::clone(){return (particle_template*)new jetparton();}

void jetparton::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<jetparton*> out; 
    std::vector<float> _pt, _eta, _phi, _e, _pdgid, _charge; 
    el -> get("pt"    , &_pt); 
    el -> get("eta"   , &_eta);
    el -> get("phi"   , &_phi); 
    el -> get("e"     , &_e); 
    el -> get("pdgid" , &_pdgid);
    el -> get("charge", &_charge); 

    std::vector<int> _index, _jet_index; 
    el -> get("index"   , &_index); 
    el -> get("JetIndex", &_jet_index); 

    std::vector<std::vector<int>> _topchild_index; 
    el -> get("TopChildIndex", &_topchild_index);  

    for (int x(0); x < _pt.size(); ++x){
        jetparton* p         = new jetparton();
        p -> pt              = _pt[x]; 
        p -> eta             = _eta[x]; 
        p -> phi             = _phi[x]; 
        p -> e               = _e[x]; 
        p -> charge          = _charge[x]; 
        p -> pdgid           = _pdgid[x]; 
    
        p -> index           = _index[x]; 
        p -> jet_index       = _jet_index[x]; 
        p -> topchild_index  = _topchild_index[x]; 
        (*prt)[std::string(p -> hash)] = p;
    }
}

jetparton::~jetparton(){}

// ============================= Electron ========================= //

electron::electron(){
    this -> type = "el"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("e"     , "_e"); 
    this -> add_leaf("charge", "_charge"); 

    this -> apply_type_prefix(); 
}

particle_template* electron::clone(){return (particle_template*)new electron();}

void electron::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<electron*> out; 
    std::vector<float> _pt, _eta, _phi, _e, _charge; 
    el -> get("pt"    , &_pt); 
    el -> get("eta"   , &_eta);
    el -> get("phi"   , &_phi); 
    el -> get("e"     , &_e); 
    el -> get("charge", &_charge); 

    for (int x(0); x < _pt.size(); ++x){
        electron* p         = new electron();
        p -> pt              = _pt[x]; 
        p -> eta             = _eta[x]; 
        p -> phi             = _phi[x]; 
        p -> e               = _e[x]; 
        p -> charge          = _charge[x]; 
        (*prt)[std::string(p -> hash)] = p;
    }
}

electron::~electron(){}

// ============================= Muon ========================= //
muon::muon(){
    this -> type = "mu"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("e"     , "_e"); 
    this -> add_leaf("charge", "_charge"); 

    this -> apply_type_prefix(); 
}

particle_template* muon::clone(){return (particle_template*)new muon();}

void muon::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<muon*> out; 
    std::vector<float> _pt, _eta, _phi, _e, _charge; 
    el -> get("pt"    , &_pt); 
    el -> get("eta"   , &_eta);
    el -> get("phi"   , &_phi); 
    el -> get("e"     , &_e); 
    el -> get("charge", &_charge); 

    for (int x(0); x < _pt.size(); ++x){
        muon* p              = new muon();
        p -> pt              = _pt[x]; 
        p -> eta             = _eta[x]; 
        p -> phi             = _phi[x]; 
        p -> e               = _e[x]; 
        p -> charge          = _charge[x]; 
        (*prt)[std::string(p -> hash)] = p;
    }
}

muon::~muon(){}
