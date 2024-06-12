#include "graphs.h"

truth_tops::truth_tops(){
    this -> name = "truth_tops";
    this -> device = "cuda"; 
}

truth_tops::~truth_tops(){}
graph_template* truth_tops::clone(){return (graph_template*)new truth_tops();}

void truth_tops::build_event(event_template* ev){
    bsm_4tops* event = (bsm_4tops*)ev; 
    this -> add_particle_nodes(&event -> Tops); 
    this -> add_graph_feature(signal(event), "signal"); 
    this -> define_topology(res_edge);
    this -> inv(signal<bool>, event); 
}
