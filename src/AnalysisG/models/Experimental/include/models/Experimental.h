#ifndef GNN_EXPERIMENTAL_H
#define GNN_EXPERIMENTAL_H
#include <templates/model_template.h>

class experimental: public model_template
{
    public:
        experimental(); 
        ~experimental();
        model_template* clone() override;
        void forward(graph_t*) override; 

        torch::Tensor message(
            torch::Tensor trk_i, torch::Tensor trk_j, torch::Tensor pmc,
            torch::Tensor hx_i, torch::Tensor hx_j
        ); 

        // Neural Network Parameters
        int _xin  = 6; 
        int _dxin = 5; 
        int _xout = 2; 
        int _hidden = 256; 

        double drop_out = 0.01; 

        // Misc
        bool is_mc = true; 

        torch::nn::Sequential* rnn_x        = nullptr; 
        torch::nn::Sequential* rnn_dx       = nullptr; 
        torch::nn::Sequential* rnn_merge    = nullptr; 
        torch::nn::Sequential* rnn_top_edge = nullptr;
        torch::nn::Sequential* rnn_res_edge = nullptr; 
        torch::nn::Sequential* mlp_ntop     = nullptr; 
        torch::nn::Sequential* mlp_sig      = nullptr; 

}; 

#endif
