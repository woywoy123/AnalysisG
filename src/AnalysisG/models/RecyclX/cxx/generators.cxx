#include <utils.h>
#include <recyclx.h>
#include <pyc/pyc.h>
#include <vector>

torch::Tensor utils::get_edge(recyclx* ml, graph_t* data){
    return utils::as_l(data -> get_edge_index(ml)); 
}

torch::Tensor utils::get_batch(recyclx* ml, graph_t* data){
    return utils::format(data -> get_batch_index(ml), -1); 
}

torch::Tensor utils::get_event(recyclx* ml, graph_t* data){
    return std::get<0>(torch::_unique(utils::get_batch(ml, data))); 
}

torch::Tensor utils::build_pmc(recyclx* ml, graph_t* data){
    torch::Tensor* pt      = data -> get_data_node("pt",     ml);
    torch::Tensor* eta     = data -> get_data_node("eta",    ml);
    torch::Tensor* phi     = data -> get_data_node("phi",    ml);
    torch::Tensor* energy  = data -> get_data_node("energy", ml);
    torch::Tensor  pmc     = pyc::transform::combined::PxPyPzE(torch::cat({*pt, *eta, *phi, *energy}, {-1})); 
    if (ml -> init){return pmc;}
    ml -> dx_nulls = ml -> dx_nulls.to(pt -> device()); 
    ml -> te_nulls = ml -> te_nulls.to(pt -> device()); 
    ml -> init = true;
    return pmc; 
}

