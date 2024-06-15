#ifndef MODEL_TEMPLATE_H
#define MODEL_TEMPLATE_H

#include <templates/enum_functions.h>
#include <templates/graph_template.h>
#include <notification/notification.h>

struct model_settings_t {
    opt_enum    e_optim; 
    std::string s_optim; 

    std::string model_name; 
    std::string model_device; 

    std::map<std::string, std::string> o_graph; 
    std::map<std::string, std::string> o_node; 
    std::map<std::string, std::string> o_edge; 

    std::vector<std::string> i_graph; 
    std::vector<std::string> i_node; 
    std::vector<std::string> i_edge; 
}; 



class model_template: 
    public notification, 
    public tools
{
    public:
        model_template();
        ~model_template();

        // set device of model
        cproperty<std::string, model_template> name; 
        cproperty<std::string, model_template> device;

        // target properties for each graph object: name - loss
        cproperty<
            std::map<std::string, std::string>, 
            std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>>
        > o_graph;
        
        cproperty<
            std::map<std::string, std::string>, 
            std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>>
        > o_node; 

        cproperty<
            std::map<std::string, std::string>, 
            std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>>
        > o_edge; 

        // requested input features
        cproperty<
            std::vector<std::string>, 
            std::map<std::string, torch::Tensor*>
        > i_graph; 

        cproperty<
            std::vector<std::string>, 
            std::map<std::string, torch::Tensor*>
        > i_node; 

        cproperty<
            std::vector<std::string>, 
            std::map<std::string, torch::Tensor*>
        > i_edge; 


        virtual model_template* clone(); 
        virtual void forward(graph_t* data); 

        void set_optimizer(std::string name); 
        void register_module(torch::nn::Sequential* data); 
        void initialize(torch::optim::Optimizer** inpt);
        bool check_features(graph_t*);  

        void clone_settings(model_settings_t* setd); 
        void import_settings(model_settings_t* setd); 

    private:
        static void set_input_features(
                std::vector<std::string>*, 
                std::map<std::string, torch::Tensor*>*
        );

        static void set_output_features(
                std::map<std::string, std::string>*, 
                std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>>*
        ); 

        static void set_device(std::string*, model_template*); 
        torch::TensorOptions* op = nullptr; 

        opt_enum                 e_optim = opt_enum::invalid_optimizer;  
        std::string              s_optim = ""; 
        torch::optim::Optimizer* m_optim = nullptr; 

        std::vector<torch::nn::Sequential*> m_data = {}; 

        std::map<std::string, torch::Tensor*> m_i_graph = {}; 
        std::map<std::string, torch::Tensor*> m_i_node = {}; 
        std::map<std::string, torch::Tensor*> m_i_edge = {}; 

        std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>> m_o_graph = {}; 
        std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>> m_o_node = {}; 
        std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>> m_o_edge = {}; 
}; 


#endif
