#ifndef RECURSIVEGRAPHNEURALNETWORK_H
#define RECURSIVEGRAPHNEURALNETWORK_H
#include <templates/model_template.h>

class recursivegraphneuralnetwork: public model_template
{
    public:
        recursivegraphneuralnetwork();
        ~recursivegraphneuralnetwork();
        model_template* clone() override;
        void forward(graph_t*) override; 

        torch::Tensor message(
            torch::Tensor trk_i, torch::Tensor trk_j,
            torch::Tensor pmc, torch::Tensor pmc_i, torch::Tensor pmc_j,
            torch::Tensor hx_i, torch::Tensor hx_j
        ); 

        torch::Tensor neutrino(
                torch::Tensor edge_index, 
                torch::Tensor pmc, 
                torch::Tensor pid, 
                torch::Tensor met_xy, 
                torch::Tensor batch, 
                std::string* hash
        ); 

        // Neural Network Parameters
        int _dx     = 26; 
        int _x      = 5; 
        int _output = 2; 
        int _rep    = 1024; 

        // Misc
        bool NuR = false; 
        bool is_mc = true; 

        torch::nn::Sequential* rnn_x      = nullptr; 
        torch::nn::Sequential* rnn_dx     = nullptr; 
        torch::nn::Sequential* rnn_merge  = nullptr; 
        torch::nn::Sequential* rnn_update = nullptr; 

        std::map<std::string, torch::Tensor> _cache = {}; 



}; 

#endif
