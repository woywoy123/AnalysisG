#include "../abstractions/abstractions.h"
#include "../abstractions/cytypes.h"

#ifndef SELECTION_H
#define SELECTION_H

namespace CyTemplate
{
    class CySelectionTemplate : public Abstraction::CyEvent
    {
        public:
            CySelectionTemplate(); 
            ~CySelectionTemplate(); 
            void Import(selection_t); 

    }; 
}

#endif
