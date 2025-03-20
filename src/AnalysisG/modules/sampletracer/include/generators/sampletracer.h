#ifndef SAMPLETRACER_GENERATOR_H
#define SAMPLETRACER_GENERATOR_H

#include <thread>
#include <container/container.h>
#include <notification/notification.h>

class sampletracer: 
    public tools, 
    public notification
{
    public:
        sampletracer();
        ~sampletracer();

        bool add_meta_data(meta* meta_, std::string filename); 
        meta* get_meta_data(std::string filename); 

        std::vector<event_template*> get_events(std::string label); 

        void fill_selections(std::map<std::string, selection_template*>* inpt); 
        bool add_event(event_template* ev, std::string label); 
        bool add_graph(graph_template* gr, std::string label);
        bool add_selection(selection_template* sel); 

        void populate_dataloader(dataloader* dl);  
        void compile_objects(int threads); 

        std::string* output_path = nullptr; 
    private:
        std::map<std::string, container> root_container = {}; 
}; 

#endif
