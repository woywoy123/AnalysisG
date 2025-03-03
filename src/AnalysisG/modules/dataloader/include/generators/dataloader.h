#ifndef DATALOADER_GENERATOR_H
#define DATALOADER_GENERATOR_H

#ifdef PYC_CUDA
#include <cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#define server true
#endif

#ifndef server
#define server false
#endif

#include <tools/tools.h>
#include <structs/property.h>
#include <structs/settings.h>
#include <notification/notification.h>
#include <templates/graph_template.h>

#include <map>
#include <random>
#include <algorithm>

class analysis; 
class model_template; 
struct model_report; 

class dataloader: 
    public notification, 
    public tools
{
    public:
        dataloader();
        ~dataloader();

        std::vector<graph_t*>* get_k_train_set(int k); 
        std::vector<graph_t*>* get_k_validation_set(int k); 
        std::vector<graph_t*>* get_test_set(); 
        std::vector<graph_t*>* build_batch(std::vector<graph_t*>* data, model_template* mdl, model_report* rep); 

        std::map<std::string, std::vector<graph_t*>>* get_inference(); 

        void generate_test_set(float percentage = 50); 
        void generate_kfold_set(int k); 
        void dump_dataset(std::string path); 
        bool restore_dataset(std::string path); 

        std::vector<graph_t*> get_random(int num = 5); 
        void extract_data(graph_t* gr); 
        void datatransfer(torch::TensorOptions* op, size_t* num_events = nullptr, size_t* prg_events = nullptr);
        bool dump_graphs(std::string path = "./", int threads = 10); 

        void restore_graphs(std::vector<std::string> paths, int threads); 
        void restore_graphs(std::string paths, int threads); 
        void start_cuda_server(); 
    
    private:
        friend class analysis;
        settings_t* setting = nullptr; 
        std::thread* cuda_mem = nullptr; 

        void cuda_memory_server(); 

        void clean_data_elements(
                std::map<std::string, int>** data_map, 
                std::vector<std::map<std::string, int>*>* loader_map
        ); 

        void shuffle(std::vector<int>* idx); 
        void shuffle(std::vector<graph_t*>* idx); 
        std::map<std::string, graph_t*>* restore_graphs_(std::vector<std::string>, int threads); 

        std::map<int, std::vector<graph_t*>*> batched_cache = {}; 

        std::map<int, std::vector<int>*> k_fold_training = {}; 
        std::map<int, std::vector<int>*> k_fold_validation = {}; 

        std::map<int, std::vector<graph_t*>*> gr_k_fold_training = {}; 
        std::map<int, std::vector<graph_t*>*> gr_k_fold_validation = {}; 

        std::vector<int>* test_set  = nullptr; 
        std::vector<int>* train_set = nullptr; 

        std::vector<std::map<std::string, int>*> truth_map_graph = {}; 
        std::vector<std::map<std::string, int>*> truth_map_node  = {}; 
        std::vector<std::map<std::string, int>*> truth_map_edge  = {}; 

        std::vector<std::map<std::string, int>*> data_map_graph = {}; 
        std::vector<std::map<std::string, int>*> data_map_node  = {}; 
        std::vector<std::map<std::string, int>*> data_map_edge  = {}; 

        std::map<std::string, int> hash_map = {}; 
        std::vector<int>*         data_index = nullptr; 
        std::vector<graph_t*>*       gr_test = nullptr; 
        std::vector<graph_t*>*      data_set = nullptr; 

        std::default_random_engine rnd{}; 
}; 

#endif
