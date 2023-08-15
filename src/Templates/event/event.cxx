#include <iostream>
#include "event.h"
#include "../tools/tools.h"

namespace CyTemplate
{
    CyEventTemplate::CyEventTemplate(){}
    CyEventTemplate::~CyEventTemplate(){}
        
    std::string CyEventTemplate::Hash()
    {
        return this -> event_hash;
    }

    void CyEventTemplate::Hash(std::string input)
    {
        if ((this -> event_hash).size() != 0)
        {
            return; 
        }

        this -> event_hash = Hashing(input);
    }
}
