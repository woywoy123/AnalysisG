#ifndef OPTIMIZER_GENERATOR_H
#define OPTIMIZER_GENERATOR_H

#include <templates/model_template.h>
#include <generators/dataloader.h>
#include <metrics/metrics.h>


struct container_model_t {
    torch::optim::Optimizer* op = nullptr; 
    model_template* model = nullptr; 
    int kfold = -1;

    void flush(){
        if (this -> op){delete this -> op;}
        if (this -> model){delete this -> model;}
    }
    
    bool is_defined(std::string name = ""){
        if (this -> op){return true;}
        this -> model -> set_optimizer(name); 
        this -> model -> initialize(&this -> op);  
        return this -> op != nullptr; 
    }
}; 

class optimizer: 
    public notification, 
    public tools
{
    public:
        optimizer();
        ~optimizer();

        void pretest_model(); 
        void define_model(model_template* _model); 
        void define_optimizer(std::string name);     
        void model_loop(std::vector<graph_t*>* data, container_model_t* model_t, int mode, int epoch); 
        void create_data_loader(std::vector<graph_template*>* input); 
        void start(); 

        int epochs = 100; 
        std::vector<int> k_folds = {}; 
        std::map<std::string, std::map<int, container_model_t>> k_model = {}; 

    private:
        dataloader* loader = nullptr;
        metrics*    metric = nullptr;  

}; 

#endif
