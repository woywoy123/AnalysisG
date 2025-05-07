#include <bsm_4tops/graph_features.h>
#include <bsm_4tops/node_features.h>
#include <bsm_4tops/edge_features.h>
#include <bsm_4tops/graphs.h>
#include <bsm_4tops/event.h>

// ---------------- GRAPH-TOPS ------------------- //
graph_tops::graph_tops(){this -> name = "graph_tops";}
graph_tops::~graph_tops(){}
graph_template* graph_tops::clone(){return new graph_tops();}

void graph_tops::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    this -> define_particle_nodes(&event -> Tops); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event , num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node");
    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et , "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 
}

// ---------------- GRAPH-TOPS ------------------- //
graph_children::graph_children(){this -> name = "graph_children";}
graph_children::~graph_children(){}
graph_template* graph_children::clone(){return new graph_children();}

void graph_children::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    this -> define_particle_nodes(&event -> Children); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, num_neutrino, "n_nu"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, num_lepton, "n_lep"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, num_tops , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_quark, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
    this -> add_node_data_feature<int, particle_template>(is_neutrino, "is_nu");
}

// ---------------- GRAPH-TRUTHJETS ------------------- //
graph_truthjets::graph_truthjets(){this -> name = "graph_truthjets";}
graph_truthjets::~graph_truthjets(){}
graph_template* graph_truthjets::clone(){return new graph_truthjets();}

void graph_truthjets::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep && !ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> TruthJets.begin(), event -> TruthJets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_neutrino, "n_nu"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_truthjets, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
    this -> add_node_data_feature<int, particle_template>(is_neutrino, "is_nu");
}

// ---------------- GRAPH-TRUTHJETS (No Neutrino) ------------------- //
graph_truthjets_nonu::graph_truthjets_nonu(){this -> name = "graph_truthjets_nonu";}
graph_truthjets_nonu::~graph_truthjets_nonu(){}
graph_template* graph_truthjets_nonu::clone(){return new graph_truthjets_nonu();}

void graph_truthjets_nonu::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep || ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> TruthJets.begin(), event -> TruthJets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_truthjets, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}


// ---------------- GRAPH-JETS ------------------- //
graph_jets::graph_jets(){this -> name = "graph_jets";}
graph_jets::~graph_jets(){}
graph_template* graph_jets::clone(){return new graph_jets();}

bool graph_jets::PreSelection(){
    bsm_4tops* evn = this -> get_event<bsm_4tops>();
    std::vector<particle_template*> leptons; 
    for (size_t x(0); x < evn -> Electrons.size(); ++x){leptons.push_back(evn -> Electrons[x]);} 
    for (size_t x(0); x < evn -> Muons.size(); ++x){leptons.push_back(evn -> Muons[x]);} 
    if (leptons.size() != 2){return false;}
    return true; //leptons[0] -> charge == leptons[1] -> charge; 
}

void graph_jets::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep && !ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et       , "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi      , "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets         , "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight     , "weight"); 
    this -> add_graph_data_feature<long  , bsm_4tops>(event, event_number     , "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}

// ---------------- GRAPH-JETS (No Neutrino) ------------------- //
graph_jets_nonu::graph_jets_nonu(){this -> name = "graph_jets_nonu";}
graph_jets_nonu::~graph_jets_nonu(){}
graph_template* graph_jets_nonu::clone(){return new graph_jets_nonu();}

void graph_jets_nonu::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep || ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}


// ---------------- GRAPH-JETS-Detector leptons (with Neutrino) ------------------- //
graph_jets_detector_lep::graph_jets_detector_lep(){this -> name = "graph_jets_detector_lep";}
graph_jets_detector_lep::~graph_jets_detector_lep(){}
graph_template* graph_jets_detector_lep::clone(){return new graph_jets_detector_lep();}

void graph_jets_detector_lep::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    nodes.insert(nodes.end(), event -> Muons.begin(), event -> Muons.end()); 
    nodes.insert(nodes.end(), event -> Electrons.begin(), event -> Electrons.end()); 
    std::vector<particle_template*> ch = event -> Children;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et  , "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi , "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets    , "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_leps    , "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number  , "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}

// ---------------- GRAPH-JETS-Detector leptons (without Neutrino) ------------------- //
graph_detector::graph_detector(){this -> name = "graph_detector";}
graph_detector::~graph_detector(){}
graph_template* graph_detector::clone(){
    graph_detector* gx = new graph_detector();
    gx -> force_match  = this -> force_match;
    gx -> num_cuda     = this -> num_cuda;
    gx -> preselection = this -> preselection; 
    return gx;
}

bool graph_detector::PreSelection(){
    bsm_4tops* evn = this -> get_event<bsm_4tops>();
    std::vector<particle_template*> leptons; 
    for (size_t x(0); x < evn -> Electrons.size(); ++x){leptons.push_back(evn -> Electrons[x]);} 
    for (size_t x(0); x < evn -> Muons.size(); ++x){leptons.push_back(evn -> Muons[x]);} 
    return leptons.size() == 2; 
}

void graph_detector::CompileEvent(){
    auto mutual =[](neutrino* n1, neutrino* n2, std::map<std::string, particle_template*>* hash_mx) -> void {
        if (!n1 || !n2){return;}
        particle_template* b1 = n1 -> bquark; particle_template* b2 = n2 -> bquark;
        particle_template* l1 = n1 -> lepton; particle_template* l2 = n2 -> lepton; 
        std::string hb1 = b1 -> hash; std::string hl1 = l1 -> hash; std::string hn1 = n1 -> hash;
        std::string hb2 = b2 -> hash; std::string hl2 = l2 -> hash; std::string hn2 = n2 -> hash;
        if (hash_mx -> count(hb1)){(*hash_mx)[hl1] = l1; (*hash_mx)[hn1] = n1;}
        if (hash_mx -> count(hl1)){(*hash_mx)[hb1] = b1; (*hash_mx)[hn1] = n1;}

        if (hash_mx -> count(hb2)){(*hash_mx)[hl2] = l2; (*hash_mx)[hn2] = n2;}
        if (hash_mx -> count(hl2)){(*hash_mx)[hb2] = b2; (*hash_mx)[hn2] = n2;}
    };

    auto assign =[](
            std::map<std::string, particle_template*>* hash_mx, bool is_res, 
            std::map<std::string, particle_template*>* out, bool has_no_lnk
    ) -> void {
        std::map<std::string, particle_template*>::iterator itx_ = hash_mx -> begin(); 
        for (; itx_ != hash_mx -> end(); ++itx_){
            std::map<std::string, particle_template*>::iterator _itx = hash_mx -> begin(); 
            for (; _itx != hash_mx -> end(); ++_itx){
                if (has_no_lnk){break;}
                if (is_res){_itx -> second -> register_parent(itx_ -> second);}
                else {_itx -> second -> register_child(itx_ -> second);}
            }
            if (out -> count(std::string(itx_ -> second -> hash))){continue;}
            (*out)[std::string(itx_ -> second -> hash)] = itx_ -> second; 
        }
    }; 

    std::string cu = "cuda:" + std::to_string(this -> threadIdx % this -> num_cuda);  

    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    std::map<int , std::map<std::string, particle_template*>> hash_map_top; 
    std::map<bool, std::map<std::string, particle_template*>> hash_map_res; 
   
    for (size_t x(0); x < event -> Muons.size(); ++x){
        muon* mx = (muon*)event -> Muons.at(x); 
        (&mx -> parents) -> clear(); (&mx -> children) -> clear(); 
        hash_map_top[mx -> top_index][std::string(mx -> hash)] = mx; 
        hash_map_res[mx -> from_res][std::string(mx -> hash)] = mx; 
    }

    for (size_t x(0); x < event -> Electrons.size(); ++x){
        electron* mx = (electron*)event -> Electrons.at(x); 
        (&mx -> parents) -> clear(); (&mx -> children) -> clear(); 
        hash_map_top[mx -> top_index][std::string(mx -> hash)] = mx; 
        hash_map_res[mx -> from_res][std::string(mx -> hash)] = mx; 
    }

    for (size_t x(0); x < event -> Jets.size(); ++x){
        jet* mx = (jet*)event -> Jets.at(x); 
        for (size_t y(0); y < mx -> top_index.size(); ++y){
            hash_map_top[mx -> top_index.at(y)][std::string(mx -> hash)] = mx; 
        }
        hash_map_res[bool(mx -> from_res)][std::string(mx -> hash)] = mx; 
        (&mx -> parents) -> clear(); (&mx -> children) -> clear(); 
    }

    std::map<std::string, particle_template*> nox = {}; 
    std::pair<particle_template*, particle_template*> nux;
    nux = this -> double_neutrino(event -> DetectorObjects, event -> met, event -> phi, cu, 172.62*1000.0, 80.385*1000.0, 1e-5, 1e-1, 50);

    std::map<int, std::map<std::string, particle_template*>>::iterator itt = hash_map_top.begin();
    for (; itt != hash_map_top.end(); ++itt){mutual((neutrino*)std::get<0>(nux), (neutrino*)std::get<1>(nux), &itt -> second);} 
    for (itt = hash_map_top.begin(); itt != hash_map_top.end(); ++itt){assign(&itt -> second, false, &nox, itt -> first < 0);} 

    std::map<bool, std::map<std::string, particle_template*>>::iterator itr = hash_map_res.begin();
    for (; itr != hash_map_res.end(); ++itr){mutual((neutrino*)std::get<0>(nux), (neutrino*)std::get<1>(nux), &itr -> second);} 
    for (itr = hash_map_res.begin(); itr != hash_map_res.end(); ++itr){assign(&itr -> second, true, &nox, itr -> first);} 
 
    std::vector<particle_template*> _nodes = {}; 
    std::map<std::string, particle_template*>::iterator itp; 
    for (itp = nox.begin(); itp != nox.end(); ++itp){_nodes.push_back(itp -> second);}
    this -> define_particle_nodes(&_nodes); 
    this -> define_topology(fulltopo); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , bsm_4tops>(event, num_tops    , "ntops"); 

    this -> add_edge_truth_feature<int, particle_template>(det_res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(det_top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et  , "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi , "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets    , "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_leps    , "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long  , bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}
