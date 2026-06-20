#ifndef GNN_URECYCLIX_H
#define GNN_URECYCLIX_H
#include <templates/model_template.h>

class recyclx;

enum class network {
    linear, layernorm, dropout,
    relu, silu, sigmoid, prelu,
    leakyrelu, tanh, invalid
}; 

struct NetOps {
    NetOps(network nt);
    NetOps(network nt, long v1); 
    std::string Name(int nxt); 
    void Apply(torch::nn::Sequential* nn, int nxt); 

    network nt = network::invalid; 
    long _l1 = -1;
    
    torch::nn::Linear*     n1 = nullptr;  
    torch::nn::LayerNorm*  n2 = nullptr;
    torch::nn::Dropout*    n3 = nullptr;  
    torch::nn::ReLU*       n4 = nullptr; 
    torch::nn::SiLU*       n5 = nullptr; 
    torch::nn::Sigmoid*    n6 = nullptr; 
    torch::nn::PReLU*      n7 = nullptr; 
    torch::nn::LeakyReLU*  n8 = nullptr; 
    torch::nn::Tanh*       n9 = nullptr; 

}; 

namespace utils {
    // ------------- FUNCTIONAL ------------- //

    torch::Tensor detach(torch::Tensor* tn); 
    torch::Tensor detach(torch::Tensor  tn); 
    
    torch::Tensor as_type(torch::Tensor* tn, torch::ScalarType xn);
    torch::Tensor as_l(torch::Tensor* tn); 
    torch::Tensor as_f(torch::Tensor* tn); 
 
    torch::Tensor get_max(torch::Tensor inpt); 
    torch::Tensor get_diff(torch::Tensor h1, torch::Tensor h2); 

    torch::Tensor get_index(torch::Tensor  h1, torch::Tensor msk); 
    torch::Tensor get_index(torch::Tensor* h1, torch::Tensor msk); 
    torch::Tensor get_index(torch::Tensor* h1, long int l); 

    torch::Tensor node_idx(torch::Tensor* batch_index); 
    
    torch::Tensor format(torch::Tensor* tn, long int l); 
    torch::Tensor format(torch::Tensor* tn, long int l1, long int l2); 
    
    torch::Tensor lzero(torch::Tensor* tn); 
    torch::Tensor lzero(torch::Tensor* tn, torch::Tensor ix); 

    torch::Tensor mzero(long int i, long int l); 
    torch::Tensor mzero(long int i, long int l, bool form); 
    
    bool isnull(torch::Tensor* inpt); 
    bool isnull(torch::Tensor  inpt); 
    
    // ------------- GENERATORS ---------- //
    torch::Tensor get_edge( recyclx* ml, graph_t* data); 
    torch::Tensor get_batch(recyclx* ml, graph_t* data); 
    torch::Tensor get_event(recyclx* ml, graph_t* data); 
   

    // -------- ACTIVATIONS --------- //
    torch::nn::Linear    make_Fx(torch::nn::Linear* tn,    long int n = -1, long int m = -1); 
    torch::nn::LayerNorm make_Fx(torch::nn::LayerNorm* tn, long int n = -1, long int m = -1); 
    torch::nn::Dropout   make_Fx(torch::nn::Dropout* tn,   long int n = -1, long int m = -1); 
    torch::nn::ReLU      make_Fx(torch::nn::ReLU* tn,      long int n = -1, long int m = -1); 
    torch::nn::SiLU      make_Fx(torch::nn::SiLU* tn,      long int n = -1, long int m = -1); 
    torch::nn::Sigmoid   make_Fx(torch::nn::Sigmoid* tn,   long int n = -1, long int m = -1); 
    torch::nn::PReLU     make_Fx(torch::nn::PReLU* tn,     long int n = -1, long int m = -1); 
    torch::nn::LeakyReLU make_Fx(torch::nn::LeakyReLU* tn, long int n = -1, long int m = -1); 
    torch::nn::Tanh      make_Fx(torch::nn::Tanh*      tn, long int n = -1, long int m = -1); 

    torch::nn::Sequential* make_Network(std::string title, std::vector<NetOps>);  
    torch::Tensor NRecode(recyclx* ml, torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn); 
    torch::Tensor build_pmc(recyclx* ml, graph_t* data); 
}


//// ------------- BUILDER ------------- //
//torch::Tensor build_pid(graph_t* data, torch::Tensor event_idx); 
//torch::Tensor build_IDX(graph_t* data, torch::Tensor src, torch::Tensor dst); 


#endif
