#ifndef CONTAINER_H
#define CONTAINER_H

#include <meta/meta.h>
#include <tools/tools.h>

#include <templates/graph_template.h>
#include <templates/event_template.h>
#include <templates/selection_template.h>

#include <generators/dataloader.h>

struct entry_t {
    std::string hash = ""; 
    std::vector<graph_t*>                m_data  = {}; 
    std::vector<graph_template*>         m_graph = {}; 
    std::vector<event_template*>         m_event = {}; 
    std::vector<selection_template*> m_selection = {}; 

    void init(); 
    void destroy(); 
    bool has_event(event_template* ev); 
    bool has_graph(graph_template* gr); 
    bool has_selection(selection_template* sel); 
    
    template <typename g>
    void destroy(std::vector<g*>* c){
        for (size_t x(0); x < c -> size(); ++x){
            delete (*c)[x]; 
            (*c)[x] = nullptr; 
        }
        std::vector<g*>().swap(*c); 
    }
}; 

class container: public tools
{
    public:
        container();
        ~container();
        void add_meta_data(meta*, std::string); 
        meta* get_meta_data(); 

        bool add_selection_template(selection_template*); 
        bool add_event_template(event_template*, std::string label); 
        bool add_graph_template(graph_template*, std::string label); 

        void fill_selections(std::map<std::string, selection_template*>* inpt); 
        void get_events(std::vector<event_template*>*, std::string label); 
        void populate_dataloader(dataloader* dl);
        void compile(size_t* len, int threadIdx); 
        size_t len(); 
        
        entry_t* add_entry(std::string hash); 

        meta*        meta_data   = nullptr; 
        std::string* filename    = nullptr; 
        std::string* output_path = nullptr; 
        std::string     label    = ""; 

        std::map<std::string, entry_t> random_access; 
        std::map<std::string, selection_template*>* merged = nullptr; 
}; 

#endif
