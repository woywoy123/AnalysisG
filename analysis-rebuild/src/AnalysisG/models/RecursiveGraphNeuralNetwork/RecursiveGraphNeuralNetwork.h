#ifndef RECURSIVEGRAPHNEURALNETWORK_H
#define RECURSIVEGRAPHNEURALNETWORK_H
#include <templates/model_template.h>

class recursivegraphneuralnetwork: public model_template
{
    public:
        recursivegraphneuralnetwork();
        ~recursivegraphneuralnetwork();
        model_template* clone() override;

        // Neural Network Parameters
        int _dx  = 2; 
        int _hidden = 128; 
        int _repeat = 1; 

        // Misc
        bool GeV = false;
        bool NuR = false; 
    
        torch::nn::Sequential* rnn_dx = nullptr; 
}; 

#endif
