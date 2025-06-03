#include <templates/model_template.h>

void model_template::set_output_features(
        std::map<std::string, std::string>* inpt, 
        std::map<std::string, std::tuple<torch::Tensor*, lossfx*>>* out_fx
){
    std::map<std::string, std::string>::iterator itx = inpt -> begin();
    for (; itx != inpt -> end(); ++itx){
        std::string o_fx = itx -> first; 
        std::string l_fx = itx -> second;
        if (out_fx -> count(o_fx)){continue;}
        (*out_fx)[o_fx] = {nullptr, new lossfx(o_fx, l_fx)}; 
    }
}

torch::Tensor* model_template::compute_loss(std::string pred, graph_enum feat){
    std::tuple<torch::Tensor*, lossfx*>* truth = nullptr; 
    torch::Tensor* prediction = nullptr; 
    switch(feat){
        case graph_enum::truth_graph: prediction = this -> m_p_graph[pred]; truth = &this -> m_o_graph[pred]; break;
        case graph_enum::truth_node:  prediction = this -> m_p_node[pred];  truth = &this -> m_o_node[pred];  break;
        case graph_enum::truth_edge:  prediction = this -> m_p_edge[pred];  truth = &this -> m_o_edge[pred];  break;
        default: break;
    }
    if (!prediction){return nullptr;}
    this -> m_p_loss[feat][pred] = std::get<1>(*truth) -> loss(prediction,  std::get<0>(*truth)); 
    return &this -> m_p_loss[feat][pred]; 
}

void model_template::train_sequence(bool train){
    if (this -> inference_mode){return;}
    if (this -> enable_anomaly){torch::autograd::AnomalyMode::set_enabled(true);}

    std::map<std::string, std::string> gr = this -> o_graph; 
    std::map<std::string, std::string> nd = this -> o_node; 
    std::map<std::string, std::string> ed = this -> o_edge; 
    if (!this -> _losses.size()){this -> _losses = std::vector<torch::Tensor*>(gr.size() + nd.size() + ed.size(), nullptr);}

    int inx = 0; 
    std::map<std::string, std::string>::iterator itr; 
    for (itr = gr.begin(); itr != gr.end(); ++itr, ++inx){this -> _losses[inx] = this -> compute_loss(itr -> first, graph_enum::truth_graph);}
    for (itr = nd.begin(); itr != nd.end(); ++itr, ++inx){this -> _losses[inx] = this -> compute_loss(itr -> first, graph_enum::truth_node);}
    for (itr = ed.begin(); itr != ed.end(); ++itr, ++inx){this -> _losses[inx] = this -> compute_loss(itr -> first, graph_enum::truth_edge);}
    if (!train || !this -> m_optim){return;}
    this -> m_optim -> zero_grad();

    inx = -1; 
    torch::Tensor lss;
    size_t lx = this -> _losses.size();
    for (size_t x(0); x < lx; ++x){
        if (!this -> _losses[x]){continue;}
        if (this -> retain_graph){
            if (x < lx-1){this -> _losses[x] -> backward({}, c10::optional<bool>(true), true);}
            else {this -> _losses[x] -> backward();}
        }
        if (inx == -1){lss = *this -> _losses[x]; inx = x;}
        else {lss += *this -> _losses[x];} 
        this -> _losses[x] = nullptr; 
    }
    if (!this -> retain_graph){lss.backward();}
    this -> m_optim -> step();
    if (lss.index({torch::isnan(lss) == false}).size({0})){return;}
    this -> failure("Found NAN in loss. Aborting");
    abort(); 
}


