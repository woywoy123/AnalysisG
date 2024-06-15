#include "graphs.h"

truth_tops::truth_tops(){
    this -> name = "truth_tops";
    this -> device = "cuda"; 
}

truth_tops::~truth_tops(){}
graph_template* truth_tops::clone(){return (graph_template*)new truth_tops();}

void truth_tops::CompileEvent(){
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



