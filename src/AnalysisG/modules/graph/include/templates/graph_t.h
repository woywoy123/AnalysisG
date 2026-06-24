
#include <c10/core/DeviceType.h>
#include <tools/tensor_cast.h>
#include <structs/enums.h>
#include <tools/tools.h>
#include <torch/torch.h>
#include <mutex> 

struct graph_hdf5; 
class graph_template; 
class dataloader; 

#ifdef PYC_CUDA
#define cu_pyc c10::kCUDA
#else
#define cu_pyc c10::kCPU
#endif

class graph_t : 
    public tools 
{
    public: 
        graph_t();
        ~graph_t(); 

        template <typename g>
        torch::Tensor* get_truth_graph(std::string _name, g* mdl){
            return this -> has_feature(graph_enum::truth_graph, _name, mdl -> device_index); 
        }
        
        template <typename g>
        torch::Tensor* get_truth_node(std::string _name, g* mdl){
            return this -> has_feature(graph_enum::truth_node, _name, mdl -> device_index); 
        }
        
        template <typename g>
        torch::Tensor* get_truth_edge(std::string _name, g* mdl){
            return this -> has_feature(graph_enum::truth_edge, _name, mdl -> device_index); 
        }
        
        template <typename g>
        torch::Tensor* get_data_graph(std::string _name, g* mdl){
            return this -> has_feature(graph_enum::data_graph, _name, mdl -> device_index); 
        }
        
        template <typename g>
        torch::Tensor* get_data_node(std::string _name, g* mdl){
            return this -> has_feature(graph_enum::data_node, _name, mdl -> device_index); 
        }
        
        template <typename g>
        torch::Tensor* get_data_edge(std::string _name, g* mdl){
            return this -> has_feature(graph_enum::data_edge, _name, mdl -> device_index); 
        }
        
        template <typename g>
        torch::Tensor* get_edge_index(g* mdl){
            return this -> has_feature(graph_enum::edge_index, "", mdl -> device_index); 
        }

        template <typename g>
        torch::Tensor* get_event_weight(g* mdl){
            return this -> has_feature(graph_enum::weight, "", mdl -> device_index); 
        }

        template <typename g>
        torch::Tensor* get_batch_index(g* mdl){
            return this -> has_feature(graph_enum::batch_index, "", mdl -> device_index); 
        }

        template <typename g>
        torch::Tensor* get_batched_events(g* mdl){
            return this -> has_feature(graph_enum::batch_events, "", mdl -> device_index); 
        }

        torch::Tensor* has_feature(graph_enum tp, std::string _name, int dev); 
        void add_truth_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_truth_node( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_truth_edge( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_data_graph( std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_data_node(  std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 
        void add_data_edge(  std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps); 

        void transfer_to_device(torch::TensorOptions* dev); 
        void _purge_all(bool data_maps = false); 

        long    num_nodes    = 0; 
        long    event_index  = 0; 
        double  event_weight = 1; 
        bool    preselection = false;

        std::vector<long> batched_events = {}; 
        std::vector<std::string*> batched_filenames = {}; 

        std::string* hash       = nullptr; 
        std::string* filename   = nullptr; 
        std::string* graph_name = nullptr; 

        c10::DeviceType device = c10::kCPU;  
        int in_use = 1; 

    private:
        friend graph_template; 
        friend dataloader; 

        bool is_owner = false; 
        std::mutex mut; 

        torch::Tensor* edge_index = nullptr; 
        std::map<std::string, int>* data_map_graph = nullptr; 
        std::map<std::string, int>* data_map_node  = nullptr;         
        std::map<std::string, int>* data_map_edge  = nullptr;         

        std::map<std::string, int>* truth_map_graph = nullptr; 
        std::map<std::string, int>* truth_map_node  = nullptr;         
        std::map<std::string, int>* truth_map_edge  = nullptr;         

        std::vector<torch::Tensor*>* data_graph = nullptr; 
        std::vector<torch::Tensor*>* data_node  = nullptr; 
        std::vector<torch::Tensor*>* data_edge  = nullptr; 
          
        std::vector<torch::Tensor*>* truth_graph = nullptr; 
        std::vector<torch::Tensor*>* truth_node  = nullptr; 
        std::vector<torch::Tensor*>* truth_edge  = nullptr;

        std::map<int, std::vector<torch::Tensor>> dev_data_graph = {}; 
        std::map<int, std::vector<torch::Tensor>> dev_data_node  = {}; 
        std::map<int, std::vector<torch::Tensor>> dev_data_edge  = {}; 

        std::map<int, std::vector<torch::Tensor>> dev_truth_graph = {}; 
        std::map<int, std::vector<torch::Tensor>> dev_truth_node  = {}; 
        std::map<int, std::vector<torch::Tensor>> dev_truth_edge  = {};

        std::map<int, torch::Tensor> dev_edge_index   = {}; 
        std::map<int, torch::Tensor> dev_batch_index  = {}; 
        std::map<int, torch::Tensor> dev_event_weight = {};
        std::map<int, torch::Tensor> dev_batched_events = {};  
        std::map<int, bool> device_index = {}; 

        void meta_serialize(std::map<std::string, int>* data, std::string* out); 
        void meta_serialize(std::vector<torch::Tensor*>* data, std::string* out); 
        void meta_serialize(torch::Tensor* data, std::string* out); 
        void serialize(graph_hdf5* m_hdf5);

        void meta_deserialize(std::map<std::string, int>* data, std::string* inpt); 
        void meta_deserialize(std::vector<torch::Tensor*>* data, std::string* inpt); 
        torch::Tensor* meta_deserialize(std::string* inpt); 
        void deserialize(graph_hdf5* m_hdf5);

        void _purge_data(std::vector<torch::Tensor*>* data); 
        void _purge_data(std::map<int, torch::Tensor*>* data); 
        void _purge_data(std::map<int, std::vector<torch::Tensor*>*>* data); 
        std::vector<torch::Tensor*>* add_content(std::map<std::string, torch::Tensor*>* inpt); 

        void _transfer_to_device(
                std::vector<torch::Tensor>* trg, 
                std::vector<torch::Tensor*>* data, 
                torch::TensorOptions* dev
        ); 

        torch::Tensor* return_any(
                std::map<std::string, int>* loc, 
                std::map<int, std::vector<torch::Tensor>>* container, 
                std::string _name, int dev_
        );
}; 



