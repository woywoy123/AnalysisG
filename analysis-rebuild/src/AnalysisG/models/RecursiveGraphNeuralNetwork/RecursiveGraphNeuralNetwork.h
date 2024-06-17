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
            torch::Tensor pmc_i, torch::Tensor pmc_j, 
            torch::Tensor pmc
        ); 



        // Neural Network Parameters
        int _dx  = 10; 
        int _x   = 7;
        int _o   = 2;

        int _hidden = 64; 
        int _repeat = 32; 

        // Misc
        bool GeV = false;
        bool NuR = false; 

        torch::nn::Sequential* rnn_x   = nullptr; 
        torch::nn::Sequential* rnn_dx  = nullptr; 
        torch::nn::Sequential* rnn_mrg = nullptr; 
}; 

#endif
