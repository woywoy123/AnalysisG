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
            CyBatch(); 
            ~CyBatch(); 
            void Import(event_t event);
            void Import(meta_t* meta);  
            batch_t Export(); 

            std::map<std::string, CyEventTemplate*> events; 
            std::map<std::string, CyGraphTemplate*> graphs; 
            std::map<std::string, CySelectionTemplate*> selections; 
           
        private: 
            bool lock_meta = false; 
            meta_t* meta; 
    }; 

    class CyROOT
    {
        public:
            CyROOT(meta_t meta);  
            ~CyROOT(); 

            root_t Export(); 
            void Import(root_t inpt); 
            void AddEvent(event_t event); 
            
            meta_t meta;
            std::map<std::string, CyBatch*> batches;
            std::map<std::string, int> n_events = {};
            std::map<std::string, int> n_graphs = {}; 
            std::map<std::string, int> n_selections = {};  


    };
}
#endif
