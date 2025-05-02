#ifndef MODEL_TEMPLATE_H
#define MODEL_TEMPLATE_H

#include <notification/notification.h>
#include <templates/graph_template.h>
#include <templates/lossfx.h>
#include <structs/settings.h>
#include <structs/model.h>

#ifdef PYC_CUDA
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraph.h>
#endif

class metrics; 
class analysis; 
class model_template; 
class metric_template;
class optimizer; 
class dataloader; 

struct graph_t; 
struct variable_t; 
struct optimizer_params_t;
struct model_report; 

class model_template: 
    public notification, 
    public tools
{
    public:
        model_template();
        virtual ~model_template();

        // set device of model
        cproperty<int, model_template> device_index;
        cproperty<std::string, model_template> name; 
        cproperty<std::string, model_template> device;

        // model state
        int kfold; 
        int epoch; 
        bool is_mc = false; 
        bool use_pkl = false; 
        bool inference_mode = false; 

        std::string model_checkpoint_path = ""; 
        std::string weight_name = "event_weight";
        std::string tree_name  = "nominal"; 

        // target properties for each graph object: name - loss
        cproperty<
            std::map<std::string, std::string>, 
            std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>
        > o_graph;
        
        cproperty<
            std::map<std::string, std::string>, 
            std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>
        > o_node; 

        cproperty<
            std::map<std::string, std::string>, 
            std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>
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
        virtual void train_sequence(bool mode); 

        void check_features(graph_t*);  
        void set_optimizer(std::string name); 
        void initialize(optimizer_params_t*);

        void clone_settings(model_settings_t* setd); 
        void import_settings(model_settings_t* setd); 

        void forward(graph_t* data, bool train); 
        void forward(std::vector<graph_t*> data, bool train); 

        void register_module(torch::nn::Sequential* data); 
        void register_module(torch::nn::Sequential* data, mlp_init weight_init); 

        void prediction_graph_feature(std::string, torch::Tensor); 
        void prediction_node_feature(std::string, torch::Tensor); 
        void prediction_edge_feature(std::string, torch::Tensor); 
        void prediction_extra(std::string, torch::Tensor); 

        torch::Tensor* compute_loss(std::string, graph_enum); 

        void evaluation_mode(bool mode = true); 

        void save_state(); 
        bool restore_state(); 

        friend struct graph_t; 
        friend struct model_report; 
        friend class metrics; 
        friend class analysis;
        friend class optimizer; 
        friend class dataloader; 
        friend class metric_template; 

    private:
        model_template* clone(int); 

        static void set_input_features(
                std::vector<std::string>*, 
                std::map<std::string, torch::Tensor*>*
        );

        static void set_output_features(
                std::map<std::string, std::string>*, 
                std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>*
        ); 


        static void get_name(std::string*, model_template*);
        static void set_name(std::string*, model_template*);

        static void get_dev_index(int*, model_template*);
        static void set_dev_index(int*, model_template*);

        template <typename G, typename F>
        void assign(std::map<std::string, G>* inpt, graph_enum mode, F* data){
            typename std::map<std::string, G>::iterator itr = inpt -> begin(); 
            for (; itr != inpt -> end(); ++itr){this -> assign_features(itr -> first, mode, data);}
        }


        static void set_device(std::string*, model_template*); 
        torch::Tensor* assign_features(std::string inpt, graph_enum type, graph_t* data); 
        torch::Tensor* assign_features(std::string inpt, graph_enum type, std::vector<graph_t*>* data); 

        void flush_outputs(); 

        lossfx*                  m_loss   = nullptr; 
        torch::TensorOptions*    m_option = nullptr; 
        torch::optim::Optimizer* m_optim  = nullptr; 
        torch::Tensor*         edge_index = nullptr; 
        bool                    m_batched = false; 
        int                  m_device_idx = -2; 

        opt_enum         e_optim = opt_enum::invalid_optimizer;  
        std::string      s_optim = ""; 
        std::string      m_name  = "model-template"; 

        std::vector<torch::nn::Sequential*> m_data = {};
        std::vector<torch::Tensor*> _losses = {};  

        std::map<std::string, torch::Tensor*> m_i_graph = {}; 
        std::map<std::string, torch::Tensor*> m_i_node = {}; 
        std::map<std::string, torch::Tensor*> m_i_edge = {}; 

        std::map<std::string, torch::Tensor*> m_p_graph = {}; 
        std::map<std::string, torch::Tensor*> m_p_node = {}; 
        std::map<std::string, torch::Tensor*> m_p_edge = {}; 
        std::map<std::string, torch::Tensor*> m_p_undef = {}; 

        std::map<std::string, std::tuple<torch::Tensor*, loss_enum>> m_o_graph = {}; 
        std::map<std::string, std::tuple<torch::Tensor*, loss_enum>> m_o_node = {}; 
        std::map<std::string, std::tuple<torch::Tensor*, loss_enum>> m_o_edge = {}; 

        std::map<graph_enum, std::map<std::string, torch::Tensor>> m_p_loss = {}; 
}; 


#endif
