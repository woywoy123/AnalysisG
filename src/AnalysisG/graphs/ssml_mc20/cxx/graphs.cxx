#include "graphs.h"
#include "graph_features.h"
#include "node_features.h"
#include "edge_features.h"

// ---------------- GRAPH-JETS ------------------- //
graph_jets::graph_jets(){this -> name = "graph_jets";}
graph_jets::~graph_jets(){}
graph_template* graph_jets::clone(){return (graph_template*)new graph_jets();}

void graph_jets::CompileEvent(){
    ssml_mc20* event = this -> get_event<ssml_mc20>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> TruthChildren;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep && !ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, ssml_mc20>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , ssml_mc20>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , ssml_mc20>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, ssml_mc20>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, num_jets, "num_jets"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long  , ssml_mc20>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}

// ---------------- GRAPH-JETS (No Neutrino) ------------------- //
graph_jets_nonu::graph_jets_nonu(){this -> name = "graph_jets_nonu";}
graph_jets_nonu::~graph_jets_nonu(){}
graph_template* graph_jets_nonu::clone(){return (graph_template*)new graph_jets_nonu();}

void graph_jets_nonu::CompileEvent(){
    ssml_mc20* event = this -> get_event<ssml_mc20>(); 
    
    std::vector<particle_template*> nodes = {}; 
    std::vector<particle_template*> ch = event -> TruthChildren;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_lep || ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, ssml_mc20>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , ssml_mc20>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , ssml_mc20>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, ssml_mc20>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, num_jets, "num_jets"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, num_children_leps, "num_leps"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long  , ssml_mc20>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
    this -> double_neutrino(); 
}


// ---------------- GRAPH-JETS-Detector leptons (with Neutrino) ------------------- //
graph_jets_detector_lep::graph_jets_detector_lep(){this -> name = "graph_jets_detector_lep";}
graph_jets_detector_lep::~graph_jets_detector_lep(){}
graph_template* graph_jets_detector_lep::clone(){return (graph_template*)new graph_jets_detector_lep();}

void graph_jets_detector_lep::CompileEvent(){
    ssml_mc20* event = this -> get_event<ssml_mc20>(); 
    
    std::vector<particle_template*> nodes = event -> Leptons; 
    std::vector<particle_template*> ch = event -> TruthChildren;
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_nu){continue;}
        nodes.push_back(ch[x]); 
    }
    nodes.insert(nodes.end(), event -> Jets.begin(), event -> Jets.end()); 
    this -> define_particle_nodes(&nodes); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, ssml_mc20>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , ssml_mc20>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , ssml_mc20>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, ssml_mc20>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, num_jets, "num_jets"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, num_leps, "num_leps"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long  , ssml_mc20>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
}

// ---------------- GRAPH-JETS-Detector leptons ------------------- //
graph_detector::graph_detector(){this -> name = "graph_detector";}
graph_detector::~graph_detector(){}
graph_template* graph_detector::clone(){return (graph_template*)new graph_detector();}

void graph_detector::CompileEvent(){
    ssml_mc20* event = this -> get_event<ssml_mc20>(); 
    
    std::vector<particle_template*> nodes = event -> Detector; 
    this -> define_particle_nodes(&nodes); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, ssml_mc20>(event, signal_event, "signal"); 
    this -> add_graph_truth_feature<int , ssml_mc20>(event, num_lepton  , "n_lep"); 
    this -> add_graph_truth_feature<int , ssml_mc20>(event, num_tops    , "ntops"); 

    this -> add_node_truth_feature<int, particle_template>(top_node, "top_node"); 
    this -> add_node_truth_feature<int, particle_template>(res_node, "res_node"); 

    this -> add_edge_truth_feature<int, particle_template>(res_edge, "res_edge"); 
    this -> add_edge_truth_feature<int, particle_template>(top_edge, "top_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<double, ssml_mc20>(event, missing_et, "met"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, missing_phi, "phi"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, num_jets, "num_jets"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, num_leps, "num_leps"); 
    this -> add_graph_data_feature<double, ssml_mc20>(event, event_weight, "weight"); 
    this -> add_graph_data_feature<long  , ssml_mc20>(event, event_number, "event_number"); 

    this -> add_node_data_feature<double, particle_template>(pt, "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta, "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi, "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 

    this -> add_node_data_feature<int, particle_template>(is_lepton, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_bquark, "is_b");
    this -> double_neutrino(); 

}
