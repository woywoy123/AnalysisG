#include <templates/model_template.h>

void model_template::set_output_features(
        std::map<std::string, std::string>* inpt, 
        std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>* out_fx
){
    std::map<std::string, std::string>::iterator itx = inpt -> begin();
    for (; itx != inpt -> end(); ++itx){
        std::string o_fx = itx -> first; 
        std::string l_fx = itx -> second;
        loss_enum loss_type = lossfx().loss_string(l_fx);
        (*out_fx)[o_fx] = {nullptr, loss_type}; 
    }
}

torch::Tensor* model_template::compute_loss(std::string pred, graph_enum feat){
    std::tuple<torch::Tensor*, loss_enum>* truth = nullptr; 
    torch::Tensor* prediction = nullptr; 
    switch(feat){
        case graph_enum::truth_graph: prediction = this -> m_p_graph[pred]; truth = &this -> m_o_graph[pred]; break;
        case graph_enum::truth_node:  prediction = this -> m_p_node[pred];  truth = &this -> m_o_node[pred]; break;
        case graph_enum::truth_edge:  prediction = this -> m_p_edge[pred];  truth = &this -> m_o_edge[pred]; break;
        default: break;
    }
    if (!prediction){return nullptr;}
    this -> m_p_loss[feat][pred] = this -> m_loss -> loss(prediction,  std::get<0>(*truth), std::get<1>(*truth)); 
    return &this -> m_p_loss[feat][pred]; 
}

void model_template::train_sequence(bool train){
    if (this -> inference_mode){return;}

    graph_enum mode; 
    std::map<std::string, std::string> gr = this -> o_graph; 
    std::map<std::string, std::string> nd = this -> o_node; 
    std::map<std::string, std::string> ed = this -> o_edge; 
    if (!this -> _losses.size()){this -> _losses = std::vector<torch::Tensor*>(gr.size() + nd.size() + ed.size(), nullptr);}

    int inx = -1; 
    mode = graph_enum::truth_graph; 
    std::map<std::string, std::string>::iterator itr; 
    for (itr = gr.begin(); itr != gr.end(); ++itr){this -> _losses[++inx] = this -> compute_loss(itr -> first, mode);}

    mode = graph_enum::truth_node; 
    for (itr = nd.begin(); itr != nd.end(); ++itr){this -> _losses[++inx] = this -> compute_loss(itr -> first, mode);}

    mode = graph_enum::truth_edge; 
    for (itr = ed.begin(); itr != ed.end(); ++itr){this -> _losses[++inx] = this -> compute_loss(itr -> first, mode);}

    if (!train){return;}
    this -> m_optim -> zero_grad();

    inx = -1; 
    torch::Tensor lss;
    for (size_t x(0); x < this -> _losses.size(); ++x){
        if (!this -> _losses[x]){continue;}
        if (inx == -1){lss = *this -> _losses[x]; inx = x;}
        else {lss += *this -> _losses[x];} 
        this -> _losses[x] = nullptr; 
    }

    lss.backward(); 
    this -> m_optim -> step();
}


