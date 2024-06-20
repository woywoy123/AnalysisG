#ifndef CONTAINER_H
#define CONTAINER_H

#include <meta/meta.h>
#include <tools/tools.h>
#include <templates/graph_template.h>
#include <templates/event_template.h>
#include <generators/dataloader.h>

struct entry_t {
    std::string* hash = nullptr; 
    meta*   meta_data = nullptr; 
    
    std::vector<event_template*>* m_event = nullptr; 
    std::vector<graph_template*>* m_graph = nullptr; 
    std::vector<graph_t*>*        m_data  = nullptr; 
    
    void init(); 
    void destroy(); 
    bool has_event(event_template* ev); 
    bool has_graph(graph_template* gr); 
    
    template <typename g>
    void destroy(std::vector<g*>* c){
        for (size_t x(0); x < c -> size(); ++x){
            delete (*c)[x]; 
            (*c)[x] = nullptr; 
        }
        c -> clear(); 
        c -> shrink_to_fit(); 
    }

}; 



class container: public tools
{
    public:
        container();
        ~container();
        void add_meta_data(meta*, std::string); 
        bool add_event_template(event_template*, std::string label); 
        bool add_graph_template(graph_template*, std::string label); 
        void get_events(std::vector<event_template*>*, std::string label); 
        void populate_dataloader(dataloader* dl);
        void compile(); 
        
    private:
        meta*       meta_data = nullptr; 
        std::string* filename = nullptr; 
        std::string* label    = nullptr; 

        std::map<std::string, int>*     hash_map = nullptr; 
        std::vector<entry_t*>*     random_access = nullptr; 
}; 

#endif
