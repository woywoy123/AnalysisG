#ifndef <model-name>_H
#define <model-name>_H
#include <templates/model_template.h>

class <model-name>: public model_template
{
    public:
        <model-name>();
        ~<model-name>();
        model_template* clone() override;
        void forward(graph_t*) override; 

        torch::nn::Sequential* example = nullptr; 
}; 

#endif
