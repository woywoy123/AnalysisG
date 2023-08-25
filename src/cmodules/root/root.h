#include "../abstractions/abstractions.h"
#include "../abstractions/cytypes.h"
#include "../event/event.h"

#ifndef ROOT_H
#define ROOT_H

using namespace CyTemplate; 

namespace SampleTracer
{
    class CyBatch
    {
        public:
            CyBatch(std::string hash); 
            ~CyBatch(); 
            void Import(const meta_t* meta); 
            void Import(const event_t* event);
            void Import(const graph_t* graph); 
            void Import(const selection_t* selection); 

            batch_t Export(); 
            std::string Hash(); 

            std::map<std::string, CyEventTemplate*> events; 
            std::map<std::string, CyGraphTemplate*> graphs; 
            std::map<std::string, CySelectionTemplate*> selections; 
           
            std::string hash; 

            const meta_t* meta;
            bool lock_meta = false;
            
            // Object Context
            bool get_event = false; 
            bool get_graph = false; 
            bool get_selection = false;

            bool m_ev = false; 
            CyEventTemplate* this_ev; 

            bool m_gr = false; 
            CyGraphTemplate* this_gr; 

            bool m_sel = false; 
            CySelectionTemplate* this_sel; 

            std::string this_tree = ""; 
    }; 

    class CyROOT
    {
        public:
            CyROOT(meta_t meta);  
            ~CyROOT(); 

            root_t Export(); 
            void Import(const root_t* inpt); 
            void AddEvent(const event_t* event); 
            
            meta_t meta;
            std::map<std::string, CyBatch*> batches;
            std::map<std::string, int> n_events = {};
            std::map<std::string, int> n_graphs = {}; 
            std::map<std::string, int> n_selections = {};  


    };
}
#endif
