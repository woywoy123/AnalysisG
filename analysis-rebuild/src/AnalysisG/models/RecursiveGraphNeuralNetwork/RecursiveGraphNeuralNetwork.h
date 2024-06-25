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
            torch::Tensor trk_i, 
            torch::Tensor trk_j,
            torch::Tensor pmc, 
            torch::Tensor H, 
            torch::Tensor H_
        ); 

        torch::Tensor aggregation(
            torch::Tensor src,
            torch::Tensor dst, 
            torch::Tensor pmc, 
            torch::Tensor H, 
            torch::Tensor H_
        ); 

        // Neural Network Parameters
        int _dx  = 3; 
        int _node = 1; 
        int _output = 2; 

        int _hidden = 32; 

        // Misc
        bool GeV = false;
        bool NuR = false; 

        torch::Tensor* _matrix = nullptr; 

        torch::nn::Sequential* rnn_dx     = nullptr; 
        torch::nn::Sequential* rnn_node   = nullptr; 
        torch::nn::Sequential* rnn_repeat = nullptr; 

}; 

#endif
