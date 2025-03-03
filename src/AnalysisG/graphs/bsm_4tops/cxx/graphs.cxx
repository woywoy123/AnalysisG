#include "graphs.h"
#include "graph_features.h"
#include "node_features.h"
#include "edge_features.h"

// ---------------- GRAPH-TOPS ------------------- //
graph_tops::graph_tops(){this -> name = "graph_tops";}
graph_tops::~graph_tops(){}
graph_template* graph_tops::clone(){
    return (graph_template*)new graph_tops();
}

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
graph_template* graph_children::clone(){return (graph_template*)new graph_children();}

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
graph_template* graph_truthjets::clone(){return (graph_template*)new graph_truthjets();}

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
graph_template* graph_truthjets_nonu::clone(){return (graph_template*)new graph_truthjets_nonu();}

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
    this -> double_neutrino(); 
}


// ---------------- GRAPH-JETS ------------------- //
graph_jets::graph_jets(){this -> name = "graph_jets";}
graph_jets::~graph_jets(){}
graph_template* graph_jets::clone(){return (graph_template*)new graph_jets();}

bool graph_jets::PreSelection(){
//    bsm_4tops* evn = this -> get_event<bsm_4tops>();
//    std::vector<particle_template*> leptons; 
//    for (size_t x(0); x < evn -> Electrons.size(); ++x){leptons.push_back(evn -> Electrons[x]);} 
//    for (size_t x(0); x < evn -> Muons.size(); ++x){leptons.push_back(evn -> Muons[x]);} 
//    if (leptons.size() != 2){return false;}
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
graph_template* graph_jets_nonu::clone(){return (graph_template*)new graph_jets_nonu();}

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
    this -> double_neutrino(); 
}


// ---------------- GRAPH-JETS-Detector leptons (with Neutrino) ------------------- //
graph_jets_detector_lep::graph_jets_detector_lep(){this -> name = "graph_jets_detector_lep";}
graph_jets_detector_lep::~graph_jets_detector_lep(){}
graph_template* graph_jets_detector_lep::clone(){return (graph_template*)new graph_jets_detector_lep();}

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
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_jets, "num_jets"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_leps, "num_leps"); 
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

// ---------------- GRAPH-JETS-Detector leptons (without Neutrino) ------------------- //
graph_detector::graph_detector(){this -> name = "graph_detector";}
graph_detector::~graph_detector(){}
graph_template* graph_detector::clone(){return (graph_template*)new graph_detector();}

void graph_detector::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    
    std::vector<particle_template*> nodes = {}; 
    nodes.insert(nodes.end(), event -> Muons.begin(), event -> Muons.end()); 
    nodes.insert(nodes.end(), event -> Electrons.begin(), event -> Electrons.end()); 
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
    this -> add_graph_data_feature<double, bsm_4tops>(event, num_leps, "num_leps"); 
    this -> add_graph_data_feature<double, bsm_4tops>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long, bsm_4tops>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<double, particle_template>(charge, "charge"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
    this -> double_neutrino(); 
}
