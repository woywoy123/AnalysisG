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
            torch::Tensor trk_i, torch::Tensor trk_j, torch::Tensor pmc,
            torch::Tensor hx_i, torch::Tensor hx_j
        ); 

        // Neural Network Parameters
        int _hidden = 1024; 
        int _xrec = 128; 

        int _xin  = 6; 
        int _xout = 2; 
        int _xtop = 5; 

        double drop_out = 0.1; 

        // Misc
        bool is_mc = true; 
        bool init  = false; 
        bool pagerank = true; 

        torch::nn::Sequential* rnn_x        = nullptr; 
        torch::nn::Sequential* rnn_dx       = nullptr; 
        torch::nn::Sequential* rnn_top_edge = nullptr;
        torch::nn::Sequential* rnn_res_edge = nullptr; 
        torch::nn::Sequential* rnn_hxx      = nullptr; 

        torch::nn::Sequential* mlp_ntop     = nullptr; 
        torch::nn::Sequential* mlp_sig      = nullptr; 
        torch::Tensor  x_nulls; 
        torch::Tensor dx_nulls; 
        torch::Tensor te_nulls; 
}; 

#endif
