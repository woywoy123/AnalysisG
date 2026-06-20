#include <recyclx.h>
#include <pyc/pyc.h>
#include <vector>

torch::Tensor utils::get_edge(recyclx* ml, graph_t* data){
    return as_l(data -> get_edge_index(ml)); 
}

torch::Tensor utils::get_batch(recyclx* ml, graph_t* data){
    return format(data -> get_batch_index(ml), -1); 
}

torch::Tensor utils::get_event(recyclx* ml, graph_t* data){
    return std::get<0>(torch::_unique(get_batch(ml, data))); 
}


