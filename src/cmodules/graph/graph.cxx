#include "../graph/graph.h"

namespace CyTemplate
{
    CyGraphTemplate::CyGraphTemplate(){}
    CyGraphTemplate::~CyGraphTemplate()
    {
        if (!this -> code_owner){return;}
        std::map<std::string, Code::CyCode*>::iterator it; 
        it = this -> edge_fx.begin();
        for (; it != this -> edge_fx.end(); ++it){delete it -> second;} 

        it = this -> node_fx.begin();
        for (; it != this -> node_fx.end(); ++it){delete it -> second;} 

        it = this -> graph_fx.begin();
        for (; it != this -> graph_fx.end(); ++it){delete it -> second;} 
        delete topo; 
    }

    bool CyGraphTemplate::operator == (CyGraphTemplate& gr)
    {
        graph_t* gr1 = &(this -> graph); 
        graph_t* gr2 = &(gr.graph); 
        return this -> is_same(gr1, gr2);  
    }

    void CyGraphTemplate::Import(graph_t gr)
    {
        this -> graph = gr; 
        this -> graph.graph = true; 
        this -> is_graph = true;
        this -> topo_hash = graph.topo_hash;  
    }

    graph_t CyGraphTemplate::Export()
    {
        graph_t gr = this -> graph; 
        std::map<std::string, Code::CyCode*>::iterator it; 

        it = this -> edge_fx.begin(); 
        for (; it != this -> edge_fx.end(); ++it){
            gr.edge_feature[it -> first] = it -> second -> hash;    
        }

        it = this -> node_fx.begin(); 
        for (; it != this -> node_fx.end(); ++it){
            gr.node_feature[it -> first] = it -> second -> hash;    
        }

        it = this -> graph_fx.begin(); 
        for (; it != this -> graph_fx.end(); ++it){
            gr.graph_feature[it -> first] = it -> second -> hash;    
        }

        if (this -> topo){gr.topo_hash = this -> topo -> hash;}
        return gr; 
    }

    void CyGraphTemplate::AddParticle(std::string hash, int p_index)
    {
        this -> graph.hash_particle[hash] = p_index; 
    }

    void CyGraphTemplate::RegisterEvent(const event_t* evnt)
    {
        graph_t* gr = &(this -> graph); 
        this -> set_event_hash(gr, evnt); 
        this -> set_event_tag(gr, evnt); 
        this -> set_event_tree(gr, evnt); 
        this -> set_event_root(gr, evnt); 
        this -> set_event_index(gr, evnt); 
    }

    void CyGraphTemplate::FullyConnected()
    {
        std::map<std::string, int>::iterator src; 
        std::map<std::string, int>::iterator dst; 
        std::map<std::string, int>* particles = &(this -> graph.hash_particle); 
        std::map<std::string, std::vector<int>>* topo = &(this -> graph.src_dst); 
        topo -> clear();
        bool self = this -> graph.self_loops; 

        src = particles -> begin(); 
        for (; src != particles -> end(); ++src){
            dst = particles -> begin(); 
            for (; dst != particles -> end(); ++dst){
                int src_ = src -> second; 
                int dst_ = dst -> second; 
                if (src_ == dst_ && !self){continue;}
                (*topo)[src -> first].push_back(dst_); 
            }
        }
    }

    std::string CyGraphTemplate::IndexToHash(int p_index)
    {
        std::string hash = this -> index_to_particle_hash[p_index]; 
        if (hash.size()){ return hash; } 
        std::map<std::string, int>* p_hashes = &(this -> graph.hash_particle); 
        std::map<std::string, int>::iterator it; 
        it = p_hashes -> begin(); 
        for (; it != p_hashes -> end(); ++it){
            if (p_index != it -> second){continue;}
            this -> index_to_particle_hash[p_index] = it -> first; 
            return it -> first; 
        }
        return hash;
    }
}

