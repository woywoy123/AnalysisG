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

        std::vector<float> pred_ntops_score  = {}; 
        std::vector<float> pred_signal_score = {}; 

        std::vector<std::vector<float>> pred_res_edge_score = {}; 
        std::vector<std::vector<float>> pred_top_edge_score = {}; 

        // -------- reconstructed particles -------------- //
        std::vector<particle_template*> reco_tops   = {}; 
        std::vector<particle_template*> reco_zprime = {};
        std::vector<particle_template*> event_particles = {}; 

        // -------- truth particles ------- //
        std::vector<particle_template*> truth_tops   = {}; 
        std::vector<particle_template*> truth_zprime = {}; 

        std::vector<int> truth_res_edge = {}; 
        std::vector<int> truth_top_edge = {}; 
        std::vector<int> truth_ntops    = {};
        std::vector<bool> truth_signal  = {}; 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    private: 
        std::map<std::string, particle_gnn*> m_event_particles = {}; 
        std::map<std::string, top_gnn*>      m_reco_tops = {}; 
        std::map<std::string, top_truth*>    m_truth_tops = {};

        std::vector<std::vector<int>>   m_edge_index = {}; 

        template <typename G>
        std::map<int, G*> sort_by_index(std::map<std::string, G*>* ipt){
            std::map<int, G*> data = {}; 
            typename std::map<std::string, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){data[int(ix -> second -> index)] = ix -> second;}
            return data; 
        }

        template <typename m, typename G, typename Gx>
        void vectorize(std::map<m, G*>* ipt, std::vector<Gx*>* vec){
            typename std::map<m, G*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){vec -> push_back(ix -> second);}
        }

        template <typename g, typename G>
        void sum(std::vector<g*>* ch, G** out){
            (*out) = new G(); 
            std::map<std::string, bool> maps; 
            for (size_t x(0); x < ch -> size(); ++x){
                if (maps[ch -> at(x) -> hash]){continue;}
                maps[ch -> at(x) -> hash] = true;
                (*out) -> iadd(ch -> at(x));
            }
        }
}; 


#endif
