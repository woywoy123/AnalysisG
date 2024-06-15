#ifndef SAMPLETRACER_GENERATOR_H
#define SAMPLETRACER_GENERATOR_H

#include <thread>
#include <io/io.h>
#include <tools/tools.h>
#include <notification/notification.h>
#include <templates/event_template.h>
#include <templates/graph_template.h>
#include "helper_templates.h"

class container 
{
    public:
        container(); 
        ~container(); 

        std::string filename = ""; 

        void flush_events();
        void threaded_compilation(); 
        void delegate_data(std::vector<graph_template*>*); 

        void register_event(std::map<std::string, event_template*>* inpt); 
        void register_event(std::map<std::string, graph_template*>* inpt); 

        std::vector<event_template*> get_event(std::string name, std::string tree); 
        std::vector<graph_template*> get_graph(std::string name, std::string tree); 
        int threads = 12; 

    private:
        std::map<std::string, int> event_hash_map = {}; 
        std::map<std::string, int> graph_hash_map = {}; 

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

        std::vector<graph_template*>* delegate_data(); 
        std::vector<event_template*> get_event(std::string type, std::string tree);  
        std::vector<graph_template*> get_graph(std::string type, std::string tree); 
        void compile(); 

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
