#include <metrics/metrics.h>
#include <TError.h>

metrics::metrics(){
    gErrorIgnoreLevel = 3000; 
}
metrics::~metrics(){
    std::map<int, analytics_t>::iterator itr = this -> registry.begin(); 
    for (; itr != this -> registry.end(); ++itr){itr -> second.purge();}
    this -> registry.clear(); 
}

model_report* metrics::register_model(model_template* mod, int kfold){
    this -> registry[kfold].model = mod; 
    this -> registry[kfold].report = new model_report(); 
    this -> registry[kfold].report -> k = kfold; 
    this -> registry[kfold].report -> run_name = this -> m_settings.run_name;

    this -> m_settings.output_path = mod -> model_checkpoint_path + "metrics/"; 
    this -> build_th1f_loss(&mod -> m_o_graph, graph_enum::truth_graph, kfold); 
    this -> build_th1f_loss(&mod -> m_o_node,  graph_enum::truth_node,  kfold); 
    this -> build_th1f_loss(&mod -> m_o_edge,  graph_enum::truth_edge,  kfold); 

    this -> build_th1f_accuracy(&mod -> m_o_graph, graph_enum::truth_graph, kfold); 
    this -> build_th1f_accuracy(&mod -> m_o_node,  graph_enum::truth_node,  kfold); 
    this -> build_th1f_accuracy(&mod -> m_o_edge,  graph_enum::truth_edge,  kfold); 

    model_report* mr = this -> registry[kfold].report;
    if (!this -> m_settings.targets.size()){return mr;}
    for (std::string var : this -> m_settings.targets){
        this -> build_th1f_mass(var, graph_enum::truth_edge, kfold); 
        this -> build_th1f_mass(var, graph_enum::data_edge , kfold); 
    }
    return mr;
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


    if (!this -> m_settings.targets.size()){return;}
    std::map<std::string, torch::Tensor*> node_feat = an -> model -> m_i_node; 
    torch::Tensor pmc = torch::cat({
            *(node_feat)[this -> m_settings.var_pt] , *(node_feat)[this -> m_settings.var_eta], 
            *(node_feat)[this -> m_settings.var_phi], *(node_feat)[this -> m_settings.var_energy]
    }, {-1}); 
    pmc = pyc::transform::combined::PxPyPzE(pmc)/1000;
    for (std::string var : this -> m_settings.targets){
        torch::Tensor* pred  = an -> model -> m_p_edge[var]; 
        torch::Tensor* truth = std::get<0>(an -> model -> m_o_edge[var]); 
        if (!truth){
            this -> warning("Invalid Target Mass Plot: " + var + ". Skipping all targets."); 
            this -> m_settings.targets = {};
            continue;
        }
        this -> add_th1f_mass(&pmc, an -> model -> edge_index, truth, pred, kfold, mode, var); 
    }
}

void metrics::dump_plots(int k){
    this -> dump_loss_plots(k);
    this -> dump_accuracy_plots(k); 

    if (!this -> m_settings.targets.size()){return;}
    this -> dump_mass_plots(k); 
}

