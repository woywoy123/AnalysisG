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
            torch::Tensor pmc
        ); 

        torch::Tensor aggregation(
            torch::Tensor edge_index, 
            torch::Tensor pmc, 
            torch::Tensor mlp_h 
        ); 

        // Neural Network Parameters
        int _dx  = 1; 
        int _x   = 6;
        int _o   = 2;

        int _hidden = 32; 
        int _repeat = 2; 

        // Misc
        bool GeV = false;
        bool NuR = false; 

        torch::nn::Sequential* rnn_mass     = nullptr; 
        torch::nn::Sequential* rnn_mass_dx  = nullptr; 
        torch::nn::Sequential* rnn_mass_mrg = nullptr; 
        torch::nn::Sequential* rnn_H = nullptr; 

        torch::Tensor* matrix = nullptr; 
        torch::Tensor* trk_i  = nullptr; 
        torch::Tensor* idx    = nullptr; 
        

}; 

#endif
