#include <metrics/<name>.h>

<name>_metric::~<name>_metric(){}
<name>_metric* <name>_metric::clone(){return new <name>_metric();}
<name>_metric::<name>_metric(){this -> name = "<name>";}

void <name>_metric::define_variables(){
    this -> register_output("event_<name>_training"   , "<variable>", &this -> <name>); 
    this -> register_output("event_<name>_validation" , "<variable>", &this -> <name>); 
    this -> register_output("event_<name>_evaluation" , "<variable>", &this -> <name>); 

    this -> register_output("global_<name>_training"  , "<variable>", &this -> global_<name>); 
    this -> register_output("global_<name>_validation", "<variable>", &this -> global_<name>); 
    this -> register_output("global_<name>_evaluation", "<variable>", &this -> global_<name>); 
}

void <name>_metric::event(){
    this -> write("event_<name>_" + this -> mode, "<variable>"  , &this -> <name>);
    this -> write("event_<name>_" + this -> mode, "<variable-2>", &this -> <name_2>, true);// <--- write is true
}

void <name>_metric::batch(){}
void <name>_metric::end(){
    this -> write("global_<name>_" + this -> mode, "<variable>", &this -> global_<name>); 
    this -> write("global_<name>_" + this -> mode, "<variable>", &this -> global_<name_2>, true); 
}


void <name>_metric::define_metric(metric_t* mtx){
    this -> mode = mtx -> mode(); 
    std::vector<long> batch_idx             = mtx -> get<std::vector<long>>( graph_enum::batch_index, "index");   
    std::vector<std::vector<int>> edge_idx  = mtx -> get<std::vector<std::vector<int>>>(  graph_enum::edge_index, "index");

    std::vector<std::vector<float>> edge_sc = mtx -> get<std::vector<std::vector<float>>>(graph_enum::pred_extra, "top_edge_score"); 
    std::vector<std::vector<int>>   edge_tr = mtx -> get<std::vector<std::vector<int>>>(  graph_enum::truth_edge, "top_edge"); 

    std::vector<std::vector<int>>   ntops_tru = mtx -> get<std::vector<std::vector<int>>>(  graph_enum::truth_graph, "ntops"); 
    std::vector<std::vector<float>> ntops_prd = mtx -> get<std::vector<std::vector<float>>>(graph_enum::pred_extra , "ntops_score");
}

