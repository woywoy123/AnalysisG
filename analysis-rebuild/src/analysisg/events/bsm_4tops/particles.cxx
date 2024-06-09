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
children::children(){
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

void children::get_from_res(bool* val, children* prt){
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

particle_template* children::clone(){return (particle_template*)new children();}

void children::build(std::map<std::string, particle_template*>* prt, element_t* el){
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
            children* p    = new children();
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

children::~children(){}

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












jet::jet(){
    this -> type = "jet"; 
    this -> add_leaf("pt" , "_pt"); 
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("e"  , "_e"); 
    this -> add_leaf("index", "_index");
    this -> apply_type_prefix(); 
}

jet::~jet(){}

parton::parton(){
    this -> type = "TJparton"; 
    this -> add_leaf("pt" , "_pt"); 
    this -> add_leaf("eta", "_eta"); 
    this -> add_leaf("phi", "_phi"); 
    this -> add_leaf("e"  , "_e"); 
    this -> add_leaf("index", "_index");
    this -> apply_type_prefix(); 
}

parton::~parton(){}







