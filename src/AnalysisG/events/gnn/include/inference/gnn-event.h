#ifndef EVENTS_GNN_EVENT_H
#define EVENTS_GNN_EVENT_H

#include <inference/gnn-particles.h>
#include <templates/event_template.h>

class gnn_event: public event_template
{
    public:
        gnn_event(); 
        ~gnn_event() override; 

        bool is_signal = false; 
        float signal_score = 0; 

        int ntops = 0; 
        float ntops_score = 0; 

        std::vector<bool> res_edge = {}; 
        std::vector<bool> top_edge = {}; 
        std::vector<float> res_score = {}; 
        std::vector<float> top_score = {}; 

        std::vector<zprime*> resonance = {};
        std::vector<top_gnn*> reco_tops = {}; 
        std::vector<top_truth*> truth_tops = {}; 
        std::vector<particle_gnn*> event_particles = {}; 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    private: 
        std::map<std::string, particle_gnn*> m_event_particles = {}; 
        std::map<std::string, top_gnn*>    m_reco_tops = {}; 
        std::map<std::string, top_truth*> m_truth_tops = {};

        std::vector<std::vector<float>> m_res_edge_score = {}; 
        std::vector<std::vector<float>> m_top_edge_score = {}; 
        std::vector<std::vector<int>>   m_edge_index = {}; 

        std::vector<std::vector<float>> m_ntops_score = {}; 
        std::vector<std::vector<float>> m_res_score   = {}; 

        template <typename G>
        std::map<int, G*> sort_by_index(std::map<std::string, G*>* ipt){
            std::map<int, G*> data = {}; 
            typename std::map<std::string, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){data[int(ix -> second -> index)] = ix -> second;}
            return data; 
        }

        template <typename m, typename G>
        void vectorize(std::map<m, G*>* ipt, std::vector<G*>* vec){
            typename std::map<m, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){vec -> push_back(ix -> second);}
        }

        template <typename g, typename G>
        void sum(std::vector<g*>* ch, G** out){
            G* prt = new G(); 
            for (size_t x(0); x < ch -> size(); ++x){prt -> iadd(ch -> at(x));}
            *out = prt; 
        }



}; 


#endif
