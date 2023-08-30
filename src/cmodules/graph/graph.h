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
            bool operator == (CyGraphTemplate& gr); 

            void Import(graph_t); 
            graph_t Export(); 

            void RegisterEvent(const event_t* evnt);
            void AddParticle(std::string, int); 
            void FullyConnected(); 
            std::string IndexToHash(int);

            bool code_owner = true; 
            std::map<std::string, Code::CyCode*> edge_fx = {}; 
            std::map<std::string, Code::CyCode*> node_fx = {}; 
            std::map<std::string, Code::CyCode*> graph_fx = {};
            std::map<std::string, Code::CyCode*> pre_sel_fx = {}; 
            Code::CyCode* topo = nullptr; 
            std::string topo_hash = ""; 

            // a hash map that is used for temporary storage to speed 
            // up the look up is used several times.
            std::map<int, std::string> index_to_particle_hash = {}; 
    }; 
}
#endif

