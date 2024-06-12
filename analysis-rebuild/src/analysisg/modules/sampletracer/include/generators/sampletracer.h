#ifndef SAMPLETRACER_GENERATOR_H
#define SAMPLETRACER_GENERATOR_H

#include <thread>
#include <io/io.h>
#include <tools/tools.h>
#include <notification/notification.h>
#include <templates/event_template.h>
#include <templates/graph_template.h>

template <typename G>
void static process_data(std::vector<G*>* ev, bool* execute){
    for (int x(0); x < ev -> size(); ++x){(*ev)[x] -> CompileEvent();}
}; 

template<typename G>
void register_object(
        std::map<std::string, G*>* inpt, 
        std::vector<std::map<std::string, G*>>* target,
        std::map<std::string, int>* hash_map
){
    typename std::map<std::string, G*>::iterator itr = inpt -> begin(); 
    for (; itr != inpt -> end(); ++itr){
        std::string _hash = itr -> second -> hash;
        std::string _type = itr -> second -> name; 
        std::string _tree = itr -> second -> tree; 
        if (!hash_map -> count(_hash)){
            (*hash_map)[_hash] = (int)target -> size(); 
            target -> push_back({}); 
        }
        int index = (*hash_map)[_hash]; 
        if ((*target)[index].count(_tree + "-" + _type)){continue;}
        (*target)[index][_tree + "-" + _type] = itr -> second;
    }
}; 

template <typename G>
void find_object(
        std::vector<G*>* trg, 
        std::vector<std::map<std::string, G*>>* src, 
        std::string name, std::string tree
){
    tools t = tools(); 
    for (size_t x(0); x < src -> size(); ++x){
        typename std::map<std::string, G*>::iterator itr = src -> at(x).begin();
        for (; itr != src -> at(x).end(); ++itr){
            if (!name.size() && !tree.size()){
                trg -> push_back(itr -> second); continue;
            }
            if (!t.has_string((std::string*)&itr -> first, name)){return;}
            if (!t.has_string((std::string*)&itr -> first, tree)){continue;}
            trg -> push_back(itr -> second); 
        }
    }
}; 

class container 
{
    public:
        container(); 
        ~container(); 

        std::string filename = ""; 

        bool threaded_compilation(); 
        void register_event(std::map<std::string, event_template*>* inpt); 
        void register_event(std::map<std::string, graph_template*>* inpt); 
        std::vector<event_template*> get_event(std::string name, std::string tree); 
        std::vector<graph_template*> get_graph(std::string name, std::string tree); 
        int threads = 12; 

    private:
        std::map<std::string, int> event_hash_map = {}; 
        std::map<std::string, int> graph_hash_map = {}; 

        std::map<std::string, std::map<std::string, bool>> event_compiled = {}; 
        std::map<std::string, std::map<std::string, bool>> graph_compiled = {}; 

        std::vector<std::map<std::string, event_template*>>* event_tree = nullptr; 
        std::vector<std::map<std::string, graph_template*>>* graph_tree = nullptr; 

}; 

class sampletracer: 
    public notification, 
    public tools
{
    public:
        sampletracer();
        ~sampletracer();

        std::vector<event_template*> get_event(std::string type, std::string tree);  
        std::vector<graph_template*> get_graph(std::string type, std::string tree); 

        template <typename G>
        void add_event(std::map<std::string, G*>* inpt){
            typename std::map<std::string, G*>::iterator itr = inpt -> begin(); 
            std::string filename = itr -> second -> filename; 
            bool hit = this -> root_container -> count(filename);
            container* data = nullptr; 
            if (!hit){
                data = new container(); 
                data -> filename = filename; 
                (*this -> root_container)[filename] = data; 
            }
            else {data = (*this -> root_container)[filename];}
            data -> register_event(inpt); 
        }; 

        std::map<std::string, container*>* root_container = nullptr; 
}; 

#endif
