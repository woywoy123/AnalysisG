#include "graphs.h"

// ---------------- GRAPH-TOPS ------------------- //
graph_tops::graph_tops(){this -> name = "graph_tops";}
graph_tops::~graph_tops(){}
graph_template* graph_tops::clone(){return (graph_template*)new graph_tops();}

void graph_tops::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    this -> define_particle_nodes(&event -> Tops); 

    // ---------------- truth ------------------- //
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal, "signal"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, ntops , "ntops"); 

    this -> add_node_truth_feature<int, top>(res_node<top>, "res_node"); 
    this -> add_edge_truth_feature<int, top>(res_edge<top>, "res_edge"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<float, bsm_4tops>(event, missingET, "met"); 
    this -> add_graph_data_feature<float, bsm_4tops>(event, missingPhi, "phi"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 

}

// ---------------- GRAPH-TOPS ------------------- //
graph_children::graph_children(){this -> name = "graph_children";}
graph_children::~graph_children(){}
graph_template* graph_children::clone(){return (graph_template*)new graph_children();}

void graph_children::CompileEvent(){
    bsm_4tops* event = this -> get_event<bsm_4tops>(); 
    this -> define_particle_nodes(&event -> Children); 

    // ---------------- truth ------------------- //
    this -> add_edge_truth_feature<int, top_children>(res_edge<top_children>, "res_edge"); 
    this -> add_node_truth_feature<int, top_children>(res_node<top_children>, "res_node"); 
    this -> add_edge_truth_feature<int, top_children>(top_edge, "top_edge"); 
    this -> add_graph_truth_feature<bool, bsm_4tops>(event, signal, "signal"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, ntops , "ntops"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, n_nu, "n_nu"); 
    this -> add_graph_truth_feature<int, bsm_4tops>(event, n_lep, "n_lep"); 

    // ---------------- data -------------------- //
    this -> add_graph_data_feature<float, bsm_4tops>(event, missingET, "met"); 
    this -> add_graph_data_feature<float, bsm_4tops>(event, missingPhi, "phi"); 

    this -> add_node_data_feature<double, particle_template>(pt    , "pt"); 
    this -> add_node_data_feature<double, particle_template>(eta   , "eta"); 
    this -> add_node_data_feature<double, particle_template>(phi   , "phi"); 
    this -> add_node_data_feature<double, particle_template>(energy, "energy"); 
    this -> add_node_data_feature<int, particle_template>(is_lep, "is_lep");
    this -> add_node_data_feature<int, particle_template>(is_b, "is_b");
    this -> add_node_data_feature<int, particle_template>(is_nu, "is_nu");
}




