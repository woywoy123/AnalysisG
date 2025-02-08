#include <pyc/tpyc.h>

TORCH_LIBRARY(graph_tensor, m){
    m.def("graph_edge_aggregation"  , &pyc::graph::edge_aggregation); 
    m.def("graph_node_aggregation"  , &pyc::graph::node_aggregation); 
}

