#ifndef GNN_GRIFT_H
#define GNN_GRIFT_H
#include <templates/model_template.h>

class grift: public model_template
{
    public:
        grift(); 
        ~grift();
        model_template* clone() override;
        void forward(graph_t*) override; 

        torch::Tensor message(
            torch::Tensor trk_i, torch::Tensor trk_j, 
            torch::Tensor* hx_i, torch::Tensor* hx_j,
            torch::Tensor pmc,   torch::Tensor* dnn
        ); 

        torch::Tensor get_value(torch::Tensor tpx); 
        torch::Tensor get_diff(torch::Tensor h1, torch::Tensor h2, torch::Tensor h3); 
        torch::Tensor expand(torch::Tensor h1, torch::Tensor msk); 

        torch::Tensor node_encode(torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn); 
        torch::Tensor build_IDX(graph_t* data, torch::Tensor src, torch::Tensor dst); 
        torch::Tensor build_pid(graph_t* data, torch::Tensor event_idx); 
        torch::Tensor build_pmc(graph_t* data); 

        bool break_loop(torch::Tensor inpt); 

        // Neural Network Parameters
        int _hidden = 1024; 
        int _xrec = 128; 

        int _xin  = 6; 
        int _xout = 2; 
        int _xtop = 5; 

        // Misc
        bool is_mc    = true; 
        bool init     = false; 

        torch::nn::Sequential* rnn_x   = nullptr; 
        torch::nn::Sequential* rnn_dx  = nullptr; 
        torch::nn::Sequential* rnn_txx = nullptr;
        torch::nn::Sequential* rnn_rxx = nullptr; 
        torch::nn::Sequential* rnn_hxx = nullptr; 

        torch::nn::Sequential* mlp_ntop = nullptr; 
        torch::nn::Sequential* mlp_sig  = nullptr; 
        torch::Tensor dx_nulls; 
        torch::Tensor te_nulls; 
}; 

#endif
