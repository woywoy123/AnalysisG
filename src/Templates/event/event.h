#ifndef EVENT_H
#define EVENT_H

namespace CyTemplate
{
    class CyEventTemplate
    {
        public: 
            CyEventTemplate(); 
            ~CyEventTemplate(); 
        
            std::string event_tree = ""; 
            std::string event_tagging = ""; 
        
            std::string implementation_name = ""; 
            std::string commit_hash = ""; 
        
            std::string pickle_string = ""; 
            signed int event_index = -1; 
            
            float weight = 1; 
            bool cached = false; 
            bool deprecated = false; 
            
            std::string Hash(); 
            void Hash(std::string input); 
         
        private: 
            std::string event_hash = ""; 
    }; 
}
#endif
