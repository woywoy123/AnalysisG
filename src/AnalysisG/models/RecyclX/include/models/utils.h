#ifndef GNN_URECYCLIX_H
#define GNN_URECYCLIX_H
#include <templates/model_template.h>

class recyclx;
namespace utils {
    // ------------- FUNCTIONAL ------------- //
    torch::Tensor detach(torch::Tensor* tn); 
    torch::Tensor detach(torch::Tensor  tn); 
    
    torch::Tensor as_type(torch::Tensor* tn, torch::ScalarType xn);
    torch::Tensor as_l(torch::Tensor* tn); 
    torch::Tensor as_f(torch::Tensor* tn); 
    
    torch::Tensor format(torch::Tensor* tn, long int l); 
    torch::Tensor format(torch::Tensor* tn, long int l1, long int l2); 
    
    torch::Tensor get_diff(torch::Tensor h1, torch::Tensor h2); 
    torch::Tensor get_max(torch::Tensor inpt, long int u = 0); 
    
    torch::Tensor lzero(torch::Tensor* tn); 
    torch::Tensor mzero(long int i, long int l); 
    torch::Tensor mzero(long int i, long int l, bool form); 
    
    bool isnull(torch::Tensor* inpt); 
    bool isnull(torch::Tensor  inpt); 
    
    torch::Tensor get_index(torch::Tensor  h1, torch::Tensor msk); 
    torch::Tensor get_index(torch::Tensor* h1, torch::Tensor msk); 
   
    torch::Tensor NRecode(torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn); 

    // ------------- GENERATORS ---------- //
    torch::Tensor get_edge( recyclx* ml, graph_t* data); 
    torch::Tensor get_batch(recyclx* ml, graph_t* data); 
    torch::Tensor get_event(recyclx* ml, graph_t* data); 
    
    // -------- ACTIVATIONS --------- //
    //torch::nn::Linear    make_FX(long int n, long int dst); 
    //torch::nn::LayerNorm make_FX(long int n, long int dst); 
    //torch::nn::PReLU     make_FX(long int n, long int dst); 
    //torch::nn::Dropout   make_FX(long int n, long int dst); 
    
    //torch::nn::ReLU      make_FX(long int n, long int dst); 
    //torch::nn::SiLU      make_FX(long int n, long int dst); 
    //torch::nn::Sigmoid   make_FX(long int n, long int dst); 

    //template <typename modl>
    //modl mdlFX(long int src = -1, long int ds = -1){return make_FX<modl>(src, ds);}


}



//// -------- UTILS ---------------- //
//torch::Tensor set_index(torch::Tensor h1, torch::Tensor msk); 
//torch::Tensor Enode(torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn); 
//torch::Tensor NRecode(torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn); 
//torch::Tensor get_index(torch::Tensor tpx); 
//
//// ------------- BUILDER ------------- //
//torch::Tensor build_pmc(graph_t* data); 
//torch::Tensor build_pid(graph_t* data, torch::Tensor event_idx); 
//torch::Tensor build_IDX(graph_t* data, torch::Tensor src, torch::Tensor dst); 


#endif
