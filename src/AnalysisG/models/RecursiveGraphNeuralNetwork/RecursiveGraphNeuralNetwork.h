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

        // Neural Network Parameters
        int _dx     = 4; 
        int _x      = 1; 
        int _output = 2; 

        int _hidden = 128; 
        int _rep = 128; 

        // Misc
        bool GeV = false;
        bool NuR = false; 

        torch::nn::Sequential* rnn_x      = nullptr; 
        torch::nn::Sequential* rnn_dx     = nullptr; 
        torch::nn::Sequential* rnn_merge  = nullptr; 
        torch::nn::Sequential* rnn_update = nullptr; 


}; 

#endif
