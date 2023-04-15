#include <string>
#include "../Headers/Tools.h"

#ifndef CYTEMPLATE_EVENT_H
#define CYTEMPLATE_EVENT_H

namespace CyTemplate
{
    class CyEvent
    {
        public:
            CyEvent(); 
            ~CyEvent(); 
            
            signed int index = -1;
            double weight = 1; 
            bool deprecated = false; 
            std::string tree = ""; 
            std::string commit_hash = ""; 
           
            std::string Hash();  
            void Hash(std::string val); 
        
        private:
            std::string _hash = ""; 

    }; 
}
#endif 

