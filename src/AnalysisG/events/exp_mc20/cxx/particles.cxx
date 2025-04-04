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

    std::vector<int> idx, _pdgid;  
    el -> get("index"  , &idx); 
    el -> get("pdgid"  , &_pdgid); 
    for (size_t x(0); x < tps.size(); ++x){
        tps[x] -> index   = idx[x]; 
        tps[x] -> charge  = ch[x]; 
        tps[x] -> pdgid   = _pdgid[x]; 
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
    this -> apply_type_prefix(); 
}

child::~child(){}
particle_template* child::clone(){return (particle_template*)new child();}
void child::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<child*> tps; 
    pmu(&tps, el); 

    std::vector<float> ch;  
    el -> get("charge", &ch); 
    
    std::vector<int> idx, _pdgid; 
    el -> get("index"  , &idx); 
    el -> get("pdgid"  , &_pdgid); 
    for (size_t x(0); x < tps.size(); ++x){
        tps[x] -> index   = idx[x]; 
        tps[x] -> charge  = ch[x]; 
        tps[x] -> pdgid   = _pdgid[x]; 
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
   
    this -> add_leaf("index"    , "_index"); 
    this -> add_leaf("top_index", "_top_index"); 
    this -> add_leaf("parton"   , "_trueflavor"); 
    this -> apply_type_prefix(); 
}

physics_detector::~physics_detector(){}
particle_template* physics_detector::clone(){return (particle_template*)new physics_detector();}
void physics_detector::build(std::map<std::string, particle_template*>* prt, element_t* el){
    auto lamb = [](physics_detector* msp){
        if (msp -> top_index.size() > 1){return true;}
        if (msp -> top_index[0] == -1){return false;}
        return true; 
    }; 

    std::vector<int> _index, parton, ch; 
    el -> get("charge", &ch); 
    el -> get("index" , &_index); 
    el -> get("parton", &parton); 

    std::vector<std::vector<int>> _top_index; 
    el -> get("top_index", &_top_index); 

    std::vector<physics_detector*> tps = {};
    tps.reserve(_top_index.size()); 
    pmu(&tps, el); 

    for (size_t x(0); x < tps.size(); ++x){
        tps[x] -> charge    = ch[x]; 
        tps[x] -> index     = _index[x]; 
        tps[x] -> top_index = _top_index[x]; 
        tps[x] -> pdgid     = parton[x]; 
        if (!lamb(tps[x])){delete tps[x]; tps[x] = nullptr;}
        else {(*prt)[std::string(tps[x] -> hash)] = tps[x];}
    }
}

physics_truth::physics_truth(){
    this -> type = "phystru"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("energy", "_e"); 
    this -> add_leaf("charge", "_charge"); 
    
    this -> add_leaf("index" , "_index"); 
    this -> add_leaf("parton", "_partontruthlabel"); 

    this -> add_leaf("type"     , "_type");
    this -> add_leaf("top_index", "_top_index"); 
    this -> apply_type_prefix(); 
}

physics_truth::~physics_truth(){}
particle_template* physics_truth::clone(){return (particle_template*)new physics_truth();}
void physics_truth::build(std::map<std::string, particle_template*>* prt, element_t* el){
    auto lamb = [](physics_truth* msp, std::vector<int>* prt){
        int pdg = msp -> pdgid; 
        pdg = std::abs(pdg);
        if (!prt -> size()){return false;}
        if (pdg == 0){return false;}
        if (pdg == 6){return false;}
        if (pdg == 24){return false;}
        if (pdg >= 11 && pdg <= 16){return false;}
        if (msp -> top_index.size() > 1){return true;}
        if (msp -> top_index[0] == -1){return false;}

        int mx(0); 
        std::map<int, int> out;
        std::map<int, int>::iterator itx; 
        for (size_t x(0); x < prt -> size(); ++x){out[std::abs((*prt)[x])] += 1;}
        for (itx = out.begin(); itx != out.end(); ++itx){
            if (itx -> second < mx){continue;}
            mx = itx -> second; pdg = itx -> first; 
        }
        msp -> pdgid = pdg; 
        return true; 
    }; 


    std::vector<physics_truth*> tps = {}; 
    pmu(&tps, el); 

    std::vector<int> _index, parton, ch; 
    el -> get("charge", &ch); 
    el -> get("index" , &_index); 
    el -> get("parton", &parton); 

    std::vector<std::vector<int>> _top_index, _type; 
    el -> get("top_index", &_top_index); 
    el -> get("type", &_type); 

    for (size_t x(0); x < tps.size(); ++x){
        tps[x] -> charge    = ch[x]; 
        tps[x] -> index     = _index[x]; 
        tps[x] -> top_index = _top_index[x]; 
        tps[x] -> pdgid     = parton[x]; 
        if (!lamb(tps[x], &_type[x])){delete tps[x]; tps[x] = nullptr;}
        else {(*prt)[std::string(tps[x] -> hash)] = tps[x];}
    }
}

jet::jet(){
    this -> type = "jet"; 
    this -> add_leaf("pt"    , "_pt"); 
    this -> add_leaf("eta"   , "_eta"); 
    this -> add_leaf("phi"   , "_phi"); 
    this -> add_leaf("energy", "_e"); 

    //this -> add_leaf("btag_65", "_isbtagged_GN2v01_65"); 
    //this -> add_leaf("btag_70", "_isbtagged_GN2v01_70"); 
    //this -> add_leaf("btag_77", "_isbtagged_GN2v01_77"); 
    //this -> add_leaf("btag_85", "_isbtagged_GN2v01_85"); 
    //this -> add_leaf("btag_90", "_isbtagged_GN2v01_90"); 

    this -> add_leaf("flav"  , "_truthflav"); 
    this -> add_leaf("parton", "_truthPartonLabel"); 
    this -> apply_type_prefix(); 
}

particle_template* jet::clone(){return (particle_template*)new jet();}

void jet::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<jet*> tps; 
    pmu(&tps, el); 

    std::vector<int> fl, prtn; 
    el -> get("flav"  , &fl); 
    el -> get("parton", &prtn); 
    for (int x(0); x < tps.size(); ++x){
        jet* p     = tps[x]; 
        p -> index = x; 
        p -> flav  = fl[x]; 
        p -> pdgid = prtn[x]; 
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
    el -> get("charge"  , &_charge);
    el -> get("delta_z0", &z0); 
    el -> get("d0sig"   , &d0);  

    std::vector<char> tight; 
    el -> get("tight", &tight); 

    std::vector<int> typ, org;
    el -> get("type"  , &typ); 
    el -> get("origin", &org); 

    for (int x(0); x < tps.size(); ++x){
        electron* p      = tps[x];
        p -> index       = x; 

        p -> charge      = _charge[x]; 
        p -> d0          = d0[x]; 
        p -> delta_z0    = z0[x]; 
        p -> is_tight    = (bool)tight[x]; 
        p -> true_type   = typ[x]; 
        p -> true_origin = org[x]; 
        p -> pdgid       = int(11 * _charge[x]); 
        (*prt)[std::string(p -> hash)] = p;
    }
}

electron::~electron(){}

// ============================= Muon ========================= //
muon::muon(){
    this -> type = "mu"; 
    this -> add_leaf("pt"      , "_pt"); 
    this -> add_leaf("eta"     , "_eta"); 
    this -> add_leaf("phi"     , "_phi"); 
    this -> add_leaf("energy"  , "_e"); 

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
    el -> get("charge"  , &_charge);
    el -> get("delta_z0", &z0); 
    el -> get("d0sig"   , &d0);  

    std::vector<char> tight; 
    el -> get("tight", &tight); 

    std::vector<int> typ, org;
    el -> get("type"  , &typ); 
    el -> get("origin", &org); 

    for (int x(0); x < tps.size(); ++x){
        muon* p       = tps[x];
        p -> index    = x; 

        p -> charge   = _charge[x]; 
        p -> d0       = d0[x]; 
        p -> delta_z0 = z0[x]; 

        p -> is_tight    = (bool)tight[x]; 
        p -> true_type   = typ[x]; 
        p -> true_origin = org[x]; 
        p -> pdgid       = int(13 * _charge[x]); 
        (*prt)[std::string(p -> hash)] = p;
    }
}

muon::~muon(){}



















