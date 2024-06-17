#ifndef DATALOADER_GENERATOR_H
#define DATALOADER_GENERATOR_H

#include <tools/tools.h>
#include <structs/property.h>
#include <notification/notification.h>
#include <templates/graph_template.h>

#include <random>
#include <algorithm>

class dataloader: 
    public notification, 
    public tools
{
    public:
        dataloader();
        ~dataloader();

        int size(); 
        cproperty<int, dataloader> index; 

        std::vector<graph_t*>* get_k_train_set(int k); 
        std::vector<graph_t*>* get_k_validation_set(int k); 
        std::vector<graph_t*>* get_test_set(); 

        void generate_test_set(float percentage = 50); 
        void generate_kfold_set(int k); 

        void add_to_collection(std::vector<graph_template*>* inpt); 
        std::vector<graph_t*> get_random(int num = 5); 

    private:
        void clean_data_elements(
                std::map<std::string, int>** data_map, 
                std::vector<std::map<std::string, int>*>* loader_map
        ); 

        void extract_data(graph_template*); 
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

        std::vector<int>*         data_index  = nullptr; 
        std::vector<graph_t*>*    data_set    = nullptr; 
        std::vector<std::string>* data_hashes = nullptr; 
        std::default_random_engine rnd{}; 
}; 

#endif
