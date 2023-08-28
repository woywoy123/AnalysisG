#include "../graph/graph.h"

namespace CyTemplate
{
    CyGraphTemplate::CyGraphTemplate(){}
    CyGraphTemplate::~CyGraphTemplate(){}
    void CyGraphTemplate::Import(graph_t gr)
    {
        this -> graph = gr; 
        this -> graph.graph = true; 
    }; 

}

