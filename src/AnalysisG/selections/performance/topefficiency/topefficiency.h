#ifndef TOPEFFICIENCY_H
#define TOPEFFICIENCY_H

#include <templates/selection_template.h>
#include <inference/gnn-event.h>

class topefficiency: public selection_template
{
    public:
        topefficiency();
        ~topefficiency() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        double pt_start  = 0; 
        double eta_start = -6; 

        double pt_end  = 1500; 
        double eta_end = 6; 

        double eta_step = 0.5; 
        double pt_step  = 100; 

        double score_start = 0; 
        double score_end   = 0.2;
        double score_step  = 0.01; 
        double score_avg   = 0.8;  

        double mass_start = 0; 
        double mass_end   = 70;
        double mass_step  = 1; 
        double mass_avg   = 170;  


        int iters(double start, double end, double step); 
        std::string region(double pt, double eta); 
        std::string region(double pt); 

        std::string decaymode(std::vector<top*> ev_tops); 
        void score_mass(
                double score_h, double score_l, double mass_h, double mass_l, 
                gnn_event* evn, int* perf_tops, std::vector<float>* out_tops, 
                std::map<std::string, int>* kin_perf = nullptr,
                std::map<std::string, int>* kin_reco = nullptr
        );  

        std::map<std::string, std::map<std::string, std::vector<float>>> p_topmass = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> t_topmass = {}; 

        std::map<std::string, std::map<std::string, std::vector<float>>> p_zmass = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> t_zmass = {}; 

        std::map<std::string, std::map<std::string, std::vector<float>>> prob_tops = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> prob_zprime = {}; 

        std::map<std::string, std::map<std::string, std::vector<int>>>   ms_cut_perf_tops = {}; 
        std::map<std::string, std::map<std::string, std::vector<int>>>   ms_cut_reco_tops = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> ms_cut_topmass   = {}; 

        std::map<std::string, std::map<std::string, std::vector<int>>> kin_truth_tops = {}; 
        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<int>>>> ms_kin_perf_tops = {}; 
        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<int>>>> ms_kin_reco_tops = {};

        std::map<std::string, std::vector<int>> n_tru_tops  = {}; 


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
