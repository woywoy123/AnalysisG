#ifndef SAMPLETRACER_GENERATOR_H
#define SAMPLETRACER_GENERATOR_H

#include <thread>
#include <container/container.h>
#include <notification/notification.h>

class sampletracer: public tools
{
    public:
        sampletracer();
        ~sampletracer();

        bool add_meta_data(meta* meta_, std::string filename); 
        std::vector<event_template*>* get_events(std::string label); 
        bool add_event(event_template* ev, std::string label); 
        bool add_graph(graph_template* gr, std::string label);
        void populate_dataloader(dataloader* dl);  
        void compile_objects(); 

    private:
        std::map<std::string, container*>* root_container = nullptr; 
}; 

#endif