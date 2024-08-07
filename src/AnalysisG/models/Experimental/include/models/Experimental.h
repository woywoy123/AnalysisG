#ifndef RECURSIVEGRAPHNEURALNETWORK_H
#define RECURSIVEGRAPHNEURALNETWORK_H
#include <templates/model_template.h>

class experimental: public model_template
{
    public:
        experimental(int rep = 26, double dpt = 0.1); 
        ~experimental();
        model_template* clone() override;
        void forward(graph_t*) override; 

        torch::Tensor message(
            torch::Tensor trk_i, torch::Tensor trk_j, torch::Tensor pmc,
            torch::Tensor hx_i, torch::Tensor hx_j
        ); 

        // Neural Network Parameters
        int _dx     = 26; 
        int _x      = 5; 
        int _output = 2; 
        int _rep    = 26; 
        int _hidden = 64; 
        double res_mass = 0; 
        double drop_out = 0.1; 

        // Misc
        bool is_mc = true; 


        torch::nn::Sequential* rnn_x      = nullptr; 
        torch::nn::Sequential* rnn_dx     = nullptr; 
        torch::nn::Sequential* rnn_merge  = nullptr; 
        torch::nn::Sequential* rnn_update = nullptr; 
        torch::nn::Sequential* rnn_exotic = nullptr; 

        torch::nn::Sequential* node_aggr_mlp = nullptr; 
        torch::nn::Sequential* ntops_mlp     = nullptr; 
        torch::nn::Sequential* exo_mlp       = nullptr; 

}; 

#endif
