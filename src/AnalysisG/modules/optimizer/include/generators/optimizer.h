#ifndef OPTIMIZER_GENERATOR_H
#define OPTIMIZER_GENERATOR_H

#include <templates/model_template.h>
#include <generators/dataloader.h>
#include <metrics/metrics.h>
#include <structs/settings.h>

class optimizer: 
    public tools,
    public notification
{
    public:
        optimizer();
        ~optimizer();

        settings_t m_settings; 
        void import_dataloader(dataloader* dl); 
        void import_model_sessions(std::tuple<model_template*, optimizer_params_t*>* models); 
        void check_model_sessions(int example_size, std::map<std::string, model_report*>* rep); 
        
        void training_loop(int k, int epoch); 
        void validation_loop(int k, int epoch);
        void evaluation_loop(int k, int epoch); 
        void launch_model(int k); 

    private:
        std::map<int, model_template*> kfold_sessions = {}; 
        std::map<std::string, model_report*> reports = {}; 
        metrics*    metric = nullptr;  
        dataloader* loader = nullptr; 

}; 

#endif
