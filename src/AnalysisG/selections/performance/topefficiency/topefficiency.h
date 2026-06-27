#ifndef TOPEFFICIENCY_H
#define TOPEFFICIENCY_H

#include <templates/selection_template.h>
#include <inference/gnn-event.h>


struct dump_t {
    void transfer(std::vector<top*>* v); 
    void transfer(std::vector<zprime*>* v); 
    void transfer(particle_gnn* v); 
    void transfer(top* v); 

    std::vector<double> pt    = {}; 
    std::vector<double> eta   = {}; 
    std::vector<double> phi   = {}; 
    std::vector<double> mass  = {};
    std::vector<double> ranks = {}; 
    std::vector<double> lowC  = {}; 
    std::vector<int> leptonic = {}; 
}; 

struct devent_t {
    double weight = 0; 
    int truth_t = 0; // truth tops
    int norm_t  = 0; // without PR
    int mask_t  = 0; // masked PR
    int nmsk_t  = 0; // no masked 
}; 

class topefficiency: public selection_template
{
    public:
        topefficiency();
        ~topefficiency(); 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        double pt_start  = 0; 
        double eta_start = 0; 

        double pt_end  = 1500; 
        double eta_end = 6; 

        double eta_step = 0.5; 
        double pt_step  = 100; 
        double score_step = 0.01; 

        int iters(double start, double end, double step); 
        std::string region(double pt, double eta); 
        std::string decaymode(std::vector<top*> ev_tops); 

        void check_matching(dump_t* dp, top* trg, std::vector<top*>* ev_tops); 
        std::tuple<dump_t*, std::string> switching(
                std::map<pagerank_e, dump_t>* prx, pagerank_e px, 
                std::string pname
        ); 
        void writing(
                std::map<pagerank_e, dump_t>* prx, pagerank_e px, 
                std::string pname
        ); 

        devent_t evnt; 
        dump_t m_particles; 
        std::map<pagerank_e, dump_t> top_r; 
        std::map<pagerank_e, dump_t> zprime_r; 
        std::map<pagerank_e, std::vector<float>> scores; 

        std::map<std::string, std::map<std::string, std::vector<float>>> p_topmass = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> t_topmass = {}; 

        std::map<std::string, std::map<std::string, std::vector<float>>> p_zmass = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> t_zmass = {}; 

        std::map<std::string, std::map<std::string, std::vector<float>>> prob_tops = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> prob_zprime = {}; 

        std::map<std::string, std::map<std::string, std::vector<float>>> t_decay_region = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> p_decay_region = {}; 

        std::map<std::string, std::map<float, int>> p_nodes = {};
        std::map<std::string, std::map<float, int>> t_nodes = {}; 

        std::map<std::string, int> n_tru_tops  = {}; 
        std::map<std::string, std::map<float, int>> n_pred_tops = {}; 
        std::map<std::string, std::map<float, int>> n_perfect_tops = {}; 

        // ROC curve variables
        std::vector<int> truth_res_edge = {}; 
        std::vector<int> truth_top_edge = {}; 

        std::vector<int> truth_ntops  = {}; 
        std::vector<int> truth_signal = {}; 

        std::vector<std::vector<float>> pred_res_edge_score = {}; 
        std::vector<std::vector<float>> pred_top_edge_score = {}; 

        std::vector<std::vector<float>> pred_ntops_score  = {}; 
        std::vector<std::vector<float>> pred_signal_score = {}; 
};

#endif
