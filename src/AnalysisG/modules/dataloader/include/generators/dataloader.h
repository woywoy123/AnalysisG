#ifndef DATALOADER_GENERATOR_H
#define DATALOADER_GENERATOR_H

#include <cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <tools/tools.h>
#include <structs/property.h>
#include <notification/notification.h>
#include <templates/graph_template.h>

#include <map>
#include <random>
#include <algorithm>

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
        std::map<std::string, std::vector<graph_t*>>* get_inference(); 

        void generate_test_set(float percentage = 50); 
        void generate_kfold_set(int k); 
        void dump_dataset(std::string path); 
        bool restore_dataset(std::string path); 

        std::vector<graph_t*> get_random(int num = 5); 
        void extract_data(graph_t* gr); 
        void datatransfer(torch::TensorOptions* op, int threads = 10);
        void start_cuda_server(); 

    private:
        void cuda_memory_server(); 
        void clean_data_elements(
                std::map<std::string, int>** data_map, 
                std::vector<std::map<std::string, int>*>* loader_map
        ); 

        void shuffle(std::vector<int>* idx); 

        std::map<int, std::vector<int>*> k_fold_training = {}; 
        std::map<int, std::vector<int>*> k_fold_validation = {}; 

        std::map<int, std::vector<graph_t*>*> gr_k_fold_training = {}; 
        std::map<int, std::vector<graph_t*>*> gr_k_fold_validation = {}; 
        std::vector<graph_t*>* gr_test = nullptr; 


        std::vector<int>* test_set  = nullptr; 
        std::vector<int>* train_set = nullptr; 

        std::vector<std::map<std::string, int>*> truth_map_graph = {}; 
        std::vector<std::map<std::string, int>*> truth_map_node  = {}; 
        std::vector<std::map<std::string, int>*> truth_map_edge  = {}; 

        std::vector<std::map<std::string, int>*> data_map_graph = {}; 
        std::vector<std::map<std::string, int>*> data_map_node  = {}; 
        std::vector<std::map<std::string, int>*> data_map_edge  = {}; 

        std::vector<int>*         data_index = nullptr; 
        std::vector<graph_t*>*    data_set   = nullptr; 
        torch::TensorOptions*     tensor_op  = nullptr; 
        std::thread*                cuda_mem = nullptr; 

        std::default_random_engine rnd{}; 
}; 

#endif
