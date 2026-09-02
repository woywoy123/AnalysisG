#ifndef AVERAGE_METRIC_H
#define AVERAGE_METRIC_H

#include <templates/metric_template.h>
#include <metrics/pagerank.h>
#include <metrics/samples.h>

class accuracy_metric: public metric_template
{
    public:
        accuracy_metric(); 
        ~accuracy_metric() override; 
        accuracy_metric* clone() override; 

        void start(metric_t* mtx) override; 
        void define_metric(metric_t* mtx) override; 
        void define_variables(metric_t* mtx) override; 

        std::vector<long>               event_index(metric_t* mtx);
        std::vector<long>               batch_index(metric_t* mtx);

        std::vector<std::vector<int>>   edge_index(metric_t* mtx);
        std::vector<std::vector<int>>   top_edge_truth(metric_t* mtx); 
        std::vector<std::vector<float>> top_edge_score(metric_t* mtx);

        std::vector<std::vector<int>>   ntops_truth(metric_t* mtx);
        std::vector<std::vector<float>> ntops_score(metric_t* mtx);



        std::vector<particle_template*> build_particles(metric_t* mtx); 
        std::vector<particle_template*> build_top(std::map<int, std::map<int, particle_template*>>* mx); 
        void pagerank(event_idx* evnt); 
        void pagerank(
                std::map<int, std::map<std::string, particle_template*>>* clust, 
                std::map<std::string, std::vector<particle_template*>>* out,
                std::map<std::string, float>* bin_out,
                std::map<int, std::map<int, float>>* bin_data
        ); 

    
        template <typename T>
        void write_var(metric_t* mtx, std::vector<particle_template*>* ptr, router_t<std::vector<T>>* rt, particle_enum val){
            typename std::vector<T> _vl = {}; 
            for (size_t x(0); x < ptr -> size(); ++x){
                int passed = 1; 
                particle_template* ptr_ = ptr -> at(x); 
                std::map<std::string, particle_template*> ch = ptr_ -> children; 
                if (ch.size() > 0){
                    std::vector<particle_template*> ch_ = this -> vectorize(&ch); 
                    int b(0), l(0), n(0); 
                    for (size_t y(0); y < ch_.size(); ++y){
                        if (ch_[y] -> is_b){b += 1;}
                        else if (ch_[y] -> is_lep){l += 1;}
                        else {n += 1;}
                    } 
                    if      (l > 0  &&  b > 0 && ch_.size() > 2){passed = +2;} // leptonic
                    else if (b >= 1 && n <= 2 && ch_.size() < 5){passed = -2;} // boosted tops merging
                    passed = (1000 * n + 100 * l + 10 * b + std::abs(passed)) * sgn(passed); 
                }
                else {passed = ( 100 * int(ptr_ -> is_lep) + 10 * int(ptr_ -> is_b) );}

                switch(val){
                    case particle_enum::pt:     _vl.push_back(double(ptr -> at(x) -> pt ));  break; 
                    case particle_enum::eta:    _vl.push_back(double(ptr -> at(x) -> eta));  break; 
                    case particle_enum::phi:    _vl.push_back(double(ptr -> at(x) -> phi));  break; 
                    case particle_enum::mass:   _vl.push_back(double(ptr -> at(x) -> mass)); break; 
                    case particle_enum::energy: _vl.push_back(double(ptr -> at(x) -> e  ));  break; 
                    case particle_enum::is_lep: _vl.push_back(passed); break; 
                    default: break;
                }
            }
            rt -> write(this, mtx, &_vl);  
        }

    private: 
        double alpha = 0.85; 
        double norm_lim = 1e-6; 
        unsigned long max_itr = 1e6; 

        router_t<int>                  dsids_idx{   "accuracy", "dsid"}; 
        router_t<int>                  ntops_prd{   "accuracy", "ntops_prd"}; 
        router_t<int>                  ntops_tru{   "accuracy", "ntops_tru"}; 
        router_t<int>                  proc_idx{    "accuracy", "process_idx"}; 
        router_t<float>                edge_prd{    "accuracy", "average_edge"}; 
        router_t<std::vector<float>>   ntops_scores{"accuracy", "ntops_scores"}; 

        router_t<std::vector<double>>  particles_pt{"accuracy",     "particle_pt"}; 
        router_t<std::vector<double>>  particles_eta{"accuracy",    "particle_eta"}; 
        router_t<std::vector<double>>  particles_phi{"accuracy",    "particle_phi"}; 
        router_t<std::vector<double>>  particles_energy{"accuracy", "particle_energy"}; 
        router_t<std::vector<int   >>  particles_chn{"accuracy",    "particle_chn"}; 

        router_t<std::vector<double>>  tops_pr_pt{"accuracy",   "tops_pr_pt"}; 
        router_t<std::vector<double>>  tops_pr_eta{"accuracy",  "tops_pr_eta"}; 
        router_t<std::vector<double>>  tops_pr_phi{"accuracy",  "tops_pr_phi"}; 
        router_t<std::vector<double>>  tops_pr_mass{"accuracy", "tops_pr_mass"}; 
        router_t<std::vector<float >>  tops_pr_scr{"accuracy",  "tops_pr_score"}; 
        router_t<std::vector<int   >>  tops_pr_chn{"accuracy",  "tops_pr_chn"}; 

        router_t<std::vector<double>>  tops_upr_pt{"accuracy",   "tops_upr_pt"}; 
        router_t<std::vector<double>>  tops_upr_eta{"accuracy",  "tops_upr_eta"}; 
        router_t<std::vector<double>>  tops_upr_phi{"accuracy",  "tops_upr_phi"}; 
        router_t<std::vector<double>>  tops_upr_mass{"accuracy", "tops_upr_mass"}; 
        router_t<std::vector<float >>  tops_upr_scr{"accuracy",  "tops_upr_score"}; 
        router_t<std::vector<int   >>  tops_upr_chn{"accuracy",  "tops_upr_chn"}; 

        router_t<std::vector<double>>  tops_nom_pt{"accuracy",   "tops_nom_pt"}; 
        router_t<std::vector<double>>  tops_nom_eta{"accuracy",  "tops_nom_eta"}; 
        router_t<std::vector<double>>  tops_nom_phi{"accuracy",  "tops_nom_phi"}; 
        router_t<std::vector<double>>  tops_nom_mass{"accuracy", "tops_nom_mass"}; 
        router_t<std::vector<float >>  tops_nom_scr{"accuracy",  "tops_nom_score"}; 
        router_t<std::vector<int   >>  tops_nom_chn{"accuracy",  "tops_nom_chn"}; 

        router_t<std::vector<double>>  tops_tru_pt{"accuracy",   "tops_truth_pt"}; 
        router_t<std::vector<double>>  tops_tru_eta{"accuracy",  "tops_truth_eta"}; 
        router_t<std::vector<double>>  tops_tru_phi{"accuracy",  "tops_truth_phi"}; 
        router_t<std::vector<double>>  tops_tru_mass{"accuracy", "tops_truth_mass"}; 
        router_t<std::vector<int   >>  tops_tru_chn{"accuracy",  "tops_truth_chn"}; 
}; 


#endif
