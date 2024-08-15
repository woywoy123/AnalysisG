#ifndef TOPEFFICIENCY_H
#define TOPEFFICIENCY_H

#include <bsm_4tops/event.h>
#include <inference/gnn-event.h>
#include <templates/selection_template.h>

class topefficiency: public selection_template
{
    public:
        topefficiency();
        ~topefficiency() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        void build_phasespace(bsm_4tops* ev);
        void build_phasespace(gnn_event* ev);


        double pt_start  = 0; 
        double eta_start = -6; 

        double pt_end  = 1500; 
        double eta_end = 6; 

        double eta_step = 0.5; 
        double pt_step  = 100; 

        int iters(double start, double end, double step); 

        std::string region(double pt, double eta); 

        std::map<std::string, std::vector<float>> truthchildren_pt_eta_topmass = {};  
        std::map<std::string, std::vector<float>> truthjets_pt_eta_topmass = {};  
        std::map<std::string, std::vector<float>> jets_pt_eta_topmass = {};  

        std::map<std::string, std::vector<float>> predicted_topmass = {}; 
        std::map<std::string, std::vector<float>> predicted_topmass_reject = {}; 

        std::map<std::string, std::vector<float>> truth_topmass = {}; 
        std::map<std::string, std::vector<float>> truth_topmass_reject = {}; 

        std::map<std::string, std::vector<float>> predicted_zprime_mass = {}; 
        std::map<std::string, std::vector<float>> truth_zprime_mass = {}; 

        std::map<std::string, std::vector<int>> n_tops_predictions = {}; 
        std::map<std::string, std::vector<int>> n_tops_real = {}; 

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
