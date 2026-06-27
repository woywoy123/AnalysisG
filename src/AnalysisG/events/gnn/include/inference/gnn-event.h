#ifndef EVENTS_GNN_EVENT_H
#define EVENTS_GNN_EVENT_H

#include <templates/event_template.h>
#include <inference/gnn-particles.h>
#include <inference/util.h>

class gnn_event: public event_template
{
    public:
        gnn_event(); 
        ~gnn_event() override; 

        void build_particles(
            std::map<int, std::map<std::string, particle_gnn*>>* prtl_map, 
            std::map<int, std::map<int, float>>* bin_map, 
            std::map<pagerank_e, std::vector<top*>>* out, 
            bool use_pr, pagerank_e scr
        ); 

        void build_particles(
            std::map<int, std::map<std::string, particle_gnn*>>* prtl_map, 
            std::map<int, std::map<int, float>>* bin_map, 
            std::map<pagerank_e, std::vector<zprime*>>* out, 
            bool use_pr, pagerank_e scr
        ); 

        std::map<std::string, float> cluster(
                std::map<int, std::map<std::string, particle_gnn*>>* clust, 
                std::map<std::string, std::vector<particle_gnn*>>* out,
                std::map<int, std::map<int, float>>* bin_data, 
                bool use_pr, pagerank_e scr
        );


        // ------- observables ------- //
        int   num_bjets = 0; 
        double num_jets = 0; 
        double num_leps = 0;  
        double met      = 0; 
        double phi      = 0; 

        // ------- MVA predictions ------ //
        int   p_ntops  = 0; 
        int   p_signal = 0; 
        float s_ntops  = 0; 
        float s_signal = 0; 

        std::vector<float> ntops_scores  = {}; 
        std::vector<float> signal_scores = {}; 

        std::vector<std::vector<float>> edge_res_scores = {}; 
        std::vector<std::vector<float>> edge_top_scores = {}; 

        std::vector<particle_gnn*> event_particles = {}; 
        
        // -------- clustered particles -------------- //
        std::map<pagerank_e, std::vector<top*>>     m_tops   = {}; 
        std::map<pagerank_e, std::vector<zprime*>>  m_zprime = {};

        std::vector<int> t_edge_res = {}; 
        std::vector<int> t_edge_top = {}; 

        int t_ntops  = 0; 
        bool t_signal = 0; 

        event_template* clone()   override; 
        void build(element_t* el) override; 
        void CompileEvent()       override; 

    private: 
        std::vector<std::vector<int>>        m_edge_index = {}; 
        std::map<std::string, particle_gnn*> m_event_particles = {}; 






}; 


#endif
