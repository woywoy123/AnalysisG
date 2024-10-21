#ifndef EVENTS_GNN_EVENT_H
#define EVENTS_GNN_EVENT_H

#include <templates/event_template.h>
#include <inference/gnn-particles.h>

template <typename g>
void reduce(element_t* el, std::string key, std::vector<g>* out){
    std::vector<std::vector<g>> tmp;
    if (!el -> get(key, &tmp)){return;}
    (*out) = tmp[0]; 
};

template <typename g>
void reduce(element_t* el, std::string key, g* out){
    std::vector<g> tmp;
    if (!el -> get(key, &tmp)){return;}
    (*out) = tmp[0]; 
};

template <typename g>
void reduce_2(element_t* el, std::string key, g* out){
    std::vector<std::vector<g>> tmp;
    if (!el -> get(key, &tmp)){return;}
    (*out) = tmp[0][0]; 
};

template <typename g>
void read(element_t* el, std::string key, std::vector<g>* out){
    if (!el -> get(key, out)){return;}
};



class gnn_event: public event_template
{
    public:
        gnn_event(); 
        ~gnn_event() override; 

        // ------- observables ------- //
        int   num_bjets = 0; 
        double num_jets = 0; 
        double num_leps = 0;  

        // ------- MVA predictions ------ //
        int   p_ntops  = 0; 
        int   p_signal = 0; 
        float s_ntops  = 0; 
        float s_signal = 0; 

        std::vector<float> ntops_scores  = {}; 
        std::vector<float> signal_scores = {}; 

        std::vector<std::vector<float>> edge_res_scores = {}; 
        std::vector<std::vector<float>> edge_top_scores = {}; 

        // -------- reconstructed particles -------------- //
        std::vector<top*>    r_tops   = {}; 
        std::vector<zprime*> r_zprime = {};
        std::vector<particle_gnn*> event_particles = {}; 

        // -------- truth particles ------- //
        std::vector<top*> t_tops   = {}; 
        std::vector<zprime*> t_zprime = {}; 

        std::vector<int> t_edge_res = {}; 
        std::vector<int> t_edge_top = {}; 

        int t_ntops  = 0; 
        bool t_signal = 0; 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    private: 
        std::vector<std::vector<int>>        m_edge_index = {}; 
        std::map<std::string, particle_gnn*> m_event_particles = {}; 

        std::map<std::string, zprime*>  m_r_zprime = {}; 
        std::map<std::string, zprime*>  m_t_zprime = {}; 
        std::map<std::string, top*>     m_r_tops = {}; 
        std::map<std::string, top*>     m_t_tops = {}; 

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
                (*out) -> register_child(ch -> at(x)); 
            }
        }
}; 


#endif
