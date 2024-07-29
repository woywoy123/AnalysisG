#include "particles.h"

top::top(){
    this -> type = "top"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("energy", "_e"); 
    this -> add_leaf("charge", "_charge"); 

    this -> add_leaf("pdgid"  , "_pdgid"); 
    this -> add_leaf("index"  , "_top_index"); 
    this -> add_leaf("barcode", "_barcode"); 
    this -> add_leaf("status" , "_status"); 
    this -> apply_type_prefix(); 
}

top::~top(){}

particle_template* top::clone(){return (particle_template*)new top();}

void top::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<top*> tps; 
    pmu(&tps, el); 
   
    std::vector<float> ch; 
    el -> get("charge", &ch); 

    std::vector<int> barcode, status, idx, pdgid;  
    el -> get("barcode", &barcode); 
    el -> get("status" , &status); 
    el -> get("index"  , &idx); 
    el -> get("pdgid"  , &pdgid); 
    for (size_t x(0); x < tps.size(); ++x){
        tps[x] -> charge = ch[x]; 
        tps[x] -> barcode = barcode[x]; 
        tps[x] -> status = status[x]; 
        tps[x] -> index = idx[x]; 
        tps[x] -> pdgid = pdgid[x]; 
        (*prt)[std::string(tps[x] -> hash)] = tps[x]; 
    }
}

child::child(){
    this -> type = "child"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("energy", "_e"); 
    this -> add_leaf("charge", "_charge"); 

    this -> add_leaf("pdgid"  , "_pdgid"); 
    this -> add_leaf("index"  , "_top_index"); 
    this -> add_leaf("barcode", "_barcode"); 
    this -> add_leaf("status" , "_status"); 
    this -> apply_type_prefix(); 
}

child::~child(){}
particle_template* child::clone(){return (particle_template*)new child();}
void child::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<child*> tps; 
    pmu(&tps, el); 
    
    std::vector<int> barcode, status, idx, pdgid;  
    el -> get("barcode", &barcode); 
    el -> get("status" , &status); 
    el -> get("index"  , &idx); 
    el -> get("pdgid"  , &pdgid); 

    std::vector<float> ch; 
    el -> get("charge", &ch); 

    for (size_t x(0); x < tps.size(); ++x){
        tps[x] -> charge  = ch[x]; 
        tps[x] -> barcode = barcode[x]; 
        tps[x] -> status  = status[x]; 
        tps[x] -> index   = idx[x]; 
        tps[x] -> pdgid   = pdgid[x]; 
        (*prt)[std::string(tps[x] -> hash)] = tps[x]; 
    }
}

physics_detector::physics_detector(){
    this -> type = "physdet"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("energy", "_e"); 
    this -> add_leaf("charge", "_charge"); 
    
    this -> add_leaf("index" , "_index"); 
    this -> add_leaf("cone"  , "_conetruthlabel"); 
    this -> add_leaf("parton", "_partontruthlabel"); 

    this -> add_leaf("type"  , "_type"); 
    this -> add_leaf("top_index", "_top_index"); 
    this -> apply_type_prefix(); 

    this -> is_jet.set_getter(get_type_jet); 
    this -> is_lepton.set_getter(get_type_lepton);
    this -> is_photon.set_getter(get_type_photon); 
}

physics_detector::~physics_detector(){}
particle_template* physics_detector::clone(){return (particle_template*)new physics_detector();}
void physics_detector::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<physics_detector*> tps; 
    pmu(&tps, el); 
   
    std::vector<int> index, parton, cone; 
    std::vector<std::vector<int>> top_index, type; 

    el -> get("index" , &index); 
    el -> get("cone"  , &cone); 
    el -> get("parton", &parton); 

    el -> get("top_index", &top_index); 
    el -> get("type"     , &type); 

    std::vector<int> ch; 
    el -> get("charge", &ch); 

    for (size_t x(0); x < tps.size(); ++x){
        tps[x] -> index = index[x]; 
        tps[x] -> parton_label = parton[x]; 
        tps[x] -> cone_label = cone[x]; 
        tps[x] -> charge = ch[x]; 

        tps[x] -> particle_type = type[x]; 
        tps[x] -> top_index = top_index[x]; 
        (*prt)[std::string(tps[x] -> hash)] = tps[x]; 
    }
}

void physics_detector::get_type_jet(bool* val, physics_detector* p){
    *val = p -> particle_type[0] == 1; 
}

void physics_detector::get_type_lepton(bool* val, physics_detector* p){
    *val = p -> particle_type[1] == 1; 
}

void physics_detector::get_type_photon(bool* val, physics_detector* p){
    *val = p -> particle_type[2] == 1; 
}

physics_truth::physics_truth(){
    this -> type = "phystru"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("energy", "_e"); 
    this -> add_leaf("charge", "_charge"); 
    
    this -> add_leaf("index" , "_index"); 
    this -> add_leaf("cone"  , "_conetruthlabel"); 
    this -> add_leaf("parton", "_partontruthlabel"); 

    this -> add_leaf("type"     , "_type"); 
    this -> add_leaf("top_index", "_top_index"); 
    this -> apply_type_prefix(); 

    this -> is_jet.set_getter(get_type_jet); 
    this -> is_lepton.set_getter(get_type_lepton);
    this -> is_photon.set_getter(get_type_photon); 
}

physics_truth::~physics_truth(){}
particle_template* physics_truth::clone(){return (particle_template*)new physics_truth();}
void physics_truth::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<physics_truth*> tps; 
    pmu(&tps, el); 
   
    std::vector<int> index, parton, cone; 
    std::vector<std::vector<int>> top_index, type; 

    el -> get("index" , &index); 
    el -> get("cone"  , &cone); 
    el -> get("parton", &parton); 

    el -> get("top_index", &top_index); 
    el -> get("type"     , &type); 

    std::vector<int> ch; 
    el -> get("charge", &ch); 

    for (size_t x(0); x < tps.size(); ++x){
        tps[x] -> index = index[x]; 
        tps[x] -> parton_label = parton[x]; 
        tps[x] -> cone_label = cone[x]; 
        tps[x] -> charge = ch[x]; 

        tps[x] -> particle_type = type[x]; 
        tps[x] -> top_index = top_index[x]; 
        (*prt)[std::string(tps[x] -> hash)] = tps[x]; 
    }
}

void physics_truth::get_type_jet(bool* val, physics_truth* p){
    *val = p -> particle_type[0] == 1; 
}

void physics_truth::get_type_lepton(bool* val, physics_truth* p){
    *val = p -> particle_type[1] == 1; 
}

void physics_truth::get_type_photon(bool* val, physics_truth* p){
    *val = p -> particle_type[2] == 1; 
}


jet::jet(){
    this -> type = "jet"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("energy", "_e"); 

    this -> add_leaf("btag_65", "_isbtagged_GN2v01_65"); 
    this -> add_leaf("btag_70", "_isbtagged_GN2v01_70"); 
    this -> add_leaf("btag_77", "_isbtagged_GN2v01_77"); 
    this -> add_leaf("btag_85", "_isbtagged_GN2v01_85"); 
    this -> add_leaf("btag_90", "_isbtagged_GN2v01_90"); 

    this -> add_leaf("flav"  , "_truthflav"); 
    this -> add_leaf("parton", "_truthPartonLabel"); 
    this -> apply_type_prefix(); 
}

particle_template* jet::clone(){return (particle_template*)new jet();}

void jet::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<jet*> tps; 
    pmu(&tps, el); 

    std::vector<char> _btag_65; 
    std::vector<char> _btag_70; 
    std::vector<char> _btag_77; 
    std::vector<char> _btag_85; 
    std::vector<char> _btag_90; 

    el -> get("btag_65", &_btag_65); 
    el -> get("btag_70", &_btag_70); 
    el -> get("btag_77", &_btag_77); 
    el -> get("btag_85", &_btag_85); 
    el -> get("btag_90", &_btag_90); 

    std::vector<int> fl, prtn; 
    el -> get("flav", &fl); 
    el -> get("parton", &prtn); 

    for (int x(0); x < tps.size(); ++x){
        jet* p = tps[x]; 
        p -> btag_65 = (bool)_btag_65[x];
        p -> btag_70 = (bool)_btag_70[x];
        p -> btag_77 = (bool)_btag_77[x];
        p -> btag_85 = (bool)_btag_85[x];
        p -> btag_90 = (bool)_btag_90[x]; 
        p -> flav    = fl[x]; 
        p -> label   = prtn[x]; 

        (*prt)[std::string(p -> hash)] = p; 
    }
}

jet::~jet(){}

electron::electron(){
    this -> type = "el"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("energy", "_e"); 

    this -> add_leaf("charge"  , "_charge"); 
    this -> add_leaf("tight"   , "_isTight"); 
    this -> add_leaf("d0sig"   , "_d0sig"); 
    this -> add_leaf("delta_z0", "_delta_z0_sintheta"); 
    this -> add_leaf("type"    , "_true_type"); 
    this -> add_leaf("origin"  , "_true_origin"); 

    this -> apply_type_prefix(); 
}

particle_template* electron::clone(){return (particle_template*)new electron();}

void electron::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<electron*> tps; 
    pmu(&tps, el); 

    std::vector<float> _charge, d0, z0; 
    el -> get("charge", &_charge);
    el -> get("delta_z0", &z0); 
    el -> get("d0sig", &d0);  

    std::vector<char> tight; 
    el -> get("tight", &tight); 

    std::vector<int> typ, org;
    el -> get("type", &typ); 
    el -> get("origin", &org); 

    for (int x(0); x < tps.size(); ++x){
        electron* p   = tps[x];
        p -> charge   = _charge[x]; 
        p -> d0       = d0[x]; 
        p -> delta_z0 = z0[x]; 

        p -> is_tight    = (bool)tight[x]; 
        p -> true_type   = typ[x]; 
        p -> true_origin = org[x]; 

        (*prt)[std::string(p -> hash)] = p;
    }
}

electron::~electron(){}

// ============================= Muon ========================= //
muon::muon(){
    this -> type = "mu"; 
    this -> add_leaf("pt"     , "_pt"); 
    this -> add_leaf("eta"    , "_eta"); 
    this -> add_leaf("phi"    , "_phi"); 
    this -> add_leaf("energy" , "_e"); 

    this -> add_leaf("charge"  , "_charge"); 
    this -> add_leaf("tight"   , "_isTight"); 
    this -> add_leaf("d0sig"   , "_d0sig"); 
    this -> add_leaf("delta_z0", "_delta_z0_sintheta"); 
    this -> add_leaf("type"    , "_true_type"); 
    this -> add_leaf("origin"  , "_true_origin"); 

    this -> apply_type_prefix(); 
}

particle_template* muon::clone(){return (particle_template*)new muon();}

void muon::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<muon*> tps; 
    pmu(&tps, el); 

    std::vector<float> _charge, d0, z0; 
    el -> get("charge", &_charge);
    el -> get("delta_z0", &z0); 
    el -> get("d0sig", &d0);  

    std::vector<char> tight; 
    el -> get("tight", &tight); 

    std::vector<int> typ, org;
    el -> get("type", &typ); 
    el -> get("origin", &org); 

    for (int x(0); x < tps.size(); ++x){
        muon* p       = tps[x];
        p -> charge   = _charge[x]; 
        p -> d0       = d0[x]; 
        p -> delta_z0 = z0[x]; 

        p -> is_tight    = (bool)tight[x]; 
        p -> true_type   = typ[x]; 
        p -> true_origin = org[x]; 

        (*prt)[std::string(p -> hash)] = p;
    }
}

muon::~muon(){}



















