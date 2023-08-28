#include "../abstractions/abstractions.h"
#include "../abstractions/cytypes.h"

#ifndef GRAPH_H
#define GRAPH_H

namespace CyTemplate
{
    class CyGraphTemplate : public Abstraction::CyEvent
    {
        public:
            CyGraphTemplate(); 
            ~CyGraphTemplate(); 
            void Import(graph_t); 

    }; 
}
#endif

