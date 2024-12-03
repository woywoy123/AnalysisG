#include <pyc/cupyc.h>

TORCH_LIBRARY(graph_cuda, m){
    m.def("graph_edge_aggregation"  , &pyc::graph::edge_aggregation); 
    m.def("graph_node_aggregation"  , &pyc::graph::node_aggregation); 
    m.def("graph_unique_aggregation", &pyc::graph::unique_aggregation); 
}
