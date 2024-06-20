#include <metrics/metrics.h>

metrics::metrics(){}
metrics::~metrics(){}

void metrics::register_model(model_template* mod, int kfold){
    this -> registry[kfold].model = mod; 
    this -> output_path = mod -> model_checkpoint_path + "metrics/"; 
    this -> build_th1f_loss(&mod -> m_o_graph, graph_enum::truth_graph, kfold); 
    this -> build_th1f_loss(&mod -> m_o_node,  graph_enum::truth_node,  kfold); 
    this -> build_th1f_loss(&mod -> m_o_edge,  graph_enum::truth_edge,  kfold); 

    std::string target = "top_edge"; 
    this -> build_th1f_mass(target, graph_enum::truth_edge, kfold); 
    this -> build_th1f_mass(target, graph_enum::data_edge , kfold); 
}


void metrics::capture(mode_enum mode, int kfold, int epoch, int smpl_len){
    analytics_t* an = &this -> registry[kfold];  
    an -> this_epoch = epoch; 

    std::map<graph_enum, std::map<std::string, torch::Tensor>> lss = an -> model -> m_p_loss; 
    this -> add_th1f_loss(&lss[graph_enum::truth_graph], &an -> loss_graph[mode], kfold, smpl_len); 
    this -> add_th1f_loss(&lss[graph_enum::truth_node],  &an -> loss_node[mode],  kfold, smpl_len); 
    this -> add_th1f_loss(&lss[graph_enum::truth_edge],  &an -> loss_edge[mode],  kfold, smpl_len); 

    std::string target = "top_edge"; 
    torch::Tensor* edge_index = an -> model -> edge_index; 
    torch::Tensor* pred  = an -> model -> m_p_edge[target]; 
    torch::Tensor* truth = std::get<0>(an -> model -> m_o_edge[target]); 
    this -> add_th1f_mass(&an -> model -> m_i_node, edge_index, truth, pred, kfold, mode); 
}

void metrics::dump_plots(){
    this -> dump_loss_plots();
    this -> dump_mass_plots(); 
}



