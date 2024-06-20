#ifndef OPTIMIZER_GENERATOR_H
#define OPTIMIZER_GENERATOR_H

#include <templates/model_template.h>
#include <generators/dataloader.h>
#include <metrics/metrics.h>

class optimizer: 
    public tools,
    public notification
{
    public:
        optimizer();
        ~optimizer();

        int epochs;
        int kfolds; 

        bool training   = true; 
        bool validation = true;
        bool evaluation = true; 
        
        int refresh = 10; 

        void import_dataloader(dataloader* dl); 
        void import_model_sessions(std::tuple<model_template*, optimizer_params_t*>* models); 
        void check_model_sessions(int example_size); 
        
        void training_loop(int k, int epoch); 
        void validation_loop(int k, int epoch);
        void evaluation_loop(int k, int epoch); 
        void launch_model(); 

    private:
        std::map<int, model_template*> kfold_sessions = {}; 
        metrics*    metric = nullptr;  
        dataloader* loader = nullptr; 





}; 

#endif
