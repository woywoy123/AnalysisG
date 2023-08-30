#include "../abstractions/cytypes.h"
#include "../selection/selection.h"
#include "../event/event.h"
#include "../graph/graph.h"

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
            std::string Hash(); 
            void Import(const meta_t*); 
            void Import(const event_t*);
            void Import(const graph_t*); 
            void Import(const selection_t*); 

            void Export(batch_t* exp); 
            batch_t Export(); 

            void Contextualize(); 
            void ApplySettings(const settings_t*); 
            void ApplyCodeHash(const std::map<std::string, Code::CyCode*>*);

            std::map<std::string, CyEventTemplate*> events; 
            std::map<std::string, CyGraphTemplate*> graphs; 
            std::map<std::string, CySelectionTemplate*> selections; 
           
            std::string hash = ""; 

            const meta_t* meta;
            bool lock_meta = false;
            
            // Object Context
            bool get_event = false; 
            bool get_graph = false; 
            bool get_selection = false;
            bool valid = false; 

            CyEventTemplate* this_ev = nullptr; 
            CyGraphTemplate* this_gr = nullptr; 
            CySelectionTemplate* this_sel = nullptr; 

            std::string this_event_name = ""; 
            std::string this_tree = "";
    }; 

    class CyROOT
    {
        public:
            CyROOT(meta_t meta);  
            ~CyROOT(); 

            static void Make(CyBatch* smpls, batch_t* exp_container){
                smpls -> Export(exp_container);     
            }; 

            root_t Export(); 
            void Import(const root_t* inpt); 
            void AddEvent(const event_t* event);
            void AddGraph(const graph_t* graph);  
            
            meta_t meta;
            std::map<std::string, CyBatch*> batches;
            std::map<std::string, int> n_events = {};
            std::map<std::string, int> n_graphs = {}; 
            std::map<std::string, int> n_selections = {};  
    };
}
#endif
