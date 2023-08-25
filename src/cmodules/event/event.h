#include "../metadata/metadata.h"
#include "../abstractions/abstractions.h"
#include "../abstractions/cytypes.h"

#ifndef EVENT_H
#define EVENT_H

namespace CyTemplate
{
    class CyEventTemplate : public Abstraction::CyEvent
    {
        public: 
            CyEventTemplate(); 
            ~CyEventTemplate();

            event_T Export(); 
            void Import(event_T); 
            void Import(event_t); 
            
            void addleaf(std::string key, std::string leaf); 
            void addbranch(std::string key, std::string branch); 
            void addtree(std::string key, std::string tree); 
            
            bool operator == (CyEventTemplate& ev);
       
            std::map<std::string, std::string> leaves = {}; 
            std::map<std::string, std::string> branches = {};  
            std::map<std::string, std::string> trees = {}; 

    }; 


    class CyGraphTemplate : public Abstraction::CyEvent
    {
        public:
            CyGraphTemplate(); 
            ~CyGraphTemplate(); 
            void Import(graph_t); 

    }; 



    class CySelectionTemplate : public Abstraction::CyEvent
    {
        public:
            CySelectionTemplate(); 
            ~CySelectionTemplate(); 
            void Import(selection_t); 

    }; 
}
#endif
