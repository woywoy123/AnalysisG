#ifndef AVERAGE_METRIC_PR_H
#define AVERAGE_METRIC_PR_H
#include <templates/particle_template.h>
#include <string>
#include <vector>

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct event_idx {
    std::vector<std::vector<int>>   edge_index     = {}; 
    std::vector<std::vector<float>> top_edge_score = {}; 
    std::vector<int>                top_edge_truth = {}; 
    std::vector<int>                top_edge_pred  = {}; 
    std::vector<float>              n_tops_score   = {}; 

    int n_tops_truth = -1; 
    int n_tops_pred  = -1; 
    int process_ix = -1; 
    std::string* file = nullptr;
    
    std::vector<particle_template*> ptx; 
    std::vector<particle_template*> reco_tops_pr; 
    std::vector<float> reco_scores_pr; 

    std::vector<particle_template*> reco_tops_upr;
    std::vector<float> reco_scores_upr; 

    std::vector<particle_template*> nominal_tops; 
    std::vector<float> reco_scores_nom; 

    std::vector<particle_template*> truth_tops; 

}; 

float edge_f1(std::vector<int>* pred, std::vector<int>* truth); 
int   get_maxidx(std::vector<float>* val); 




#endif
