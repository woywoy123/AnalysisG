#ifndef RECURSIVEGRAPHNEURALNETWORK_H
#define RECURSIVEGRAPHNEURALNETWORK_H
#include <templates/model_template.h>

class recursivegraphneuralnetwork: public model_template
{
    public:
        recursivegraphneuralnetwork(int rep = 1024, double dpt = 0.1); 
        ~recursivegraphneuralnetwork();
        model_template* clone() override;
        void forward(graph_t*) override; 

        torch::Tensor message(
            torch::Tensor trk_i, torch::Tensor trk_j,
            torch::Tensor pmc, torch::Tensor pmc_i, torch::Tensor pmc_j,
            torch::Tensor hx_i, torch::Tensor hx_j
        ); 

        // Neural Network Parameters
        int _dx     = 26; 
        int _x      = 5; 
        int _output = 2; 
        int _rep    = 1024; 
        double res_mass = 0; 
        double drop_out = 0.1; 

        // Misc
        bool is_mc = true; 

        torch::nn::Sequential* rnn_x         = nullptr; 
        torch::nn::Sequential* rnn_dx        = nullptr; 
        torch::nn::Sequential* rnn_merge     = nullptr; 
        torch::nn::Sequential* rnn_update    = nullptr; 
        torch::nn::Sequential* exotic_mlp    = nullptr; 
        torch::nn::Sequential* node_aggr_mlp = nullptr; 
        torch::nn::Sequential* ntops_mlp     = nullptr; 
        torch::nn::Sequential* exo_mlp       = nullptr; 

}; 

#endif
