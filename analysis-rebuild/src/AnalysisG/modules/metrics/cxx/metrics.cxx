#include <transform/cartesian-cuda.h>
#include <metrics/metrics.h>

metrics::metrics(){}
metrics::~metrics(){}

void metrics::register_model(model_template* mod, int kfold){
    this -> registry[kfold].model = mod; 
    this -> output_path = mod -> model_checkpoint_path + "metrics/"; 
    this -> build_th1f_loss(&mod -> m_o_graph, graph_enum::truth_graph, kfold); 
    this -> build_th1f_loss(&mod -> m_o_node,  graph_enum::truth_node,  kfold); 
    this -> build_th1f_loss(&mod -> m_o_edge,  graph_enum::truth_edge,  kfold); 

    this -> build_th1f_accuracy(&mod -> m_o_graph, graph_enum::truth_graph, kfold); 
    this -> build_th1f_accuracy(&mod -> m_o_node,  graph_enum::truth_node,  kfold); 
    this -> build_th1f_accuracy(&mod -> m_o_edge,  graph_enum::truth_edge,  kfold); 

    if (!this -> targets.size()){return;}
    for (std::string var : this -> targets){
        this -> build_th1f_mass(var, graph_enum::truth_edge, kfold); 
        this -> build_th1f_mass(var, graph_enum::data_edge , kfold); 
    }
}

void metrics::capture(mode_enum mode, int kfold, int epoch, int smpl_len){
    analytics_t* an = &this -> registry[kfold];  
    an -> this_epoch = epoch; 

    std::map<graph_enum, std::map<std::string, torch::Tensor>> lss = an -> model -> m_p_loss; 
    this -> add_th1f_loss(&lss[graph_enum::truth_graph], &an -> loss_graph[mode], kfold, smpl_len); 
    this -> add_th1f_loss(&lss[graph_enum::truth_node],  &an -> loss_node[mode],  kfold, smpl_len); 
    this -> add_th1f_loss(&lss[graph_enum::truth_edge],  &an -> loss_edge[mode],  kfold, smpl_len); 

    std::map<std::string, torch::Tensor>::iterator itr; 
    itr = lss[graph_enum::truth_graph].begin(); 
    for (; itr != lss[graph_enum::truth_graph].end(); ++itr){
        torch::Tensor* pred = an -> model -> m_p_graph[itr -> first]; 
        torch::Tensor* tru  = std::get<0>(an -> model -> m_o_graph[itr -> first]);
        this -> add_th1f_accuracy(pred, tru, an -> accuracy_graph[mode][itr -> first], kfold, smpl_len);     
    } 

    itr = lss[graph_enum::truth_node].begin(); 
    for (; itr != lss[graph_enum::truth_node].end(); ++itr){
        torch::Tensor* pred = an -> model -> m_p_node[itr -> first]; 
        torch::Tensor* tru  = std::get<0>(an -> model -> m_o_node[itr -> first]);
        this -> add_th1f_accuracy(pred, tru, an -> accuracy_node[mode][itr -> first], kfold, smpl_len);     
    } 

    itr = lss[graph_enum::truth_edge].begin(); 
    for (; itr != lss[graph_enum::truth_edge].end(); ++itr){
        torch::Tensor* pred = an -> model -> m_p_edge[itr -> first]; 
        torch::Tensor* tru  = std::get<0>(an -> model -> m_o_edge[itr -> first]);
        this -> add_th1f_accuracy(pred, tru, an -> accuracy_edge[mode][itr -> first], kfold, smpl_len);     
    } 


    if (!this -> targets.size()){return;}
    std::map<std::string, torch::Tensor*> node_feat = an -> model -> m_i_node; 
    torch::Tensor pmc = torch::cat({
            *(node_feat)[this -> var_pt] , *(node_feat)[this -> var_eta], 
            *(node_feat)[this -> var_phi], *(node_feat)[this -> var_energy]
    }, {-1}); 
    pmc = transform::cuda::PxPyPzE(pmc)/1000;
    for (std::string var : this -> targets){
        torch::Tensor* pred  = an -> model -> m_p_edge[var]; 
        torch::Tensor* truth = std::get<0>(an -> model -> m_o_edge[var]); 
        this -> add_th1f_mass(&pmc, an -> model -> edge_index, truth, pred, kfold, mode, var); 
    }
}

void metrics::dump_plots(){
    this -> dump_loss_plots();
    this -> dump_accuracy_plots(); 

    if (!this -> targets.size()){return;}
    this -> dump_mass_plots(); 
}



