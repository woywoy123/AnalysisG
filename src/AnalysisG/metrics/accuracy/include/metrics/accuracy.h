#ifndef AVERAGE_METRIC_H
#define AVERAGE_METRIC_H

#include <templates/metric_template.h>
#include <metrics/pagerank.h>
#include <unordered_map>

class accuracy_metric: public metric_template
{
    public:
        accuracy_metric(); 
        ~accuracy_metric() override; 
        accuracy_metric* clone() override; 

        void define_metric(metric_t* mtx) override; 
        void define_variables(metric_t* mtx) override; 
        void end() override; 

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



    private: 
        double alpha = 0.85; 
        double norm_lim = 1e-6; 
        long max_itr = 1e6; 

        router_t<int>                ntops_prd{"event_accuracy", "ntops_prd"}; 
        router_t<int>                ntops_tru{"event_accuracy", "ntops_tru"}; 
        router_t<int>                 proc_idx{"event_accuracy", "process_idx"}; 
        router_t<float>               edge_prd{"event_accuracy", "edge"}; 
        router_t<std::vector<float>> ntops_scores{"event_accuracy", "ntops_scores"}; 
}; 





struct cdata_t {
    int kfold = -1;
    std::vector<int> ntops_truth = {}; 
    std::vector<std::vector<double>> ntop_score = {};
    std::map<int, std::vector<double>> ntop_edge_accuracy = {};
    std::map<int, std::map<int, std::vector<double>>> ntru_npred_matrix = {}; 
}; 

struct cmodel_t {
    std::map<int, std::map<int, cdata_t>> evaluation_kfold_data = {}; 
    std::map<int, std::map<int, cdata_t>> validation_kfold_data = {}; 
    std::map<int, std::map<int, cdata_t>> training_kfold_data   = {}; 
}; 

class collector: public tools
{
    public:
        collector(); 
        ~collector();

        cdata_t* get_mode(std::string model, std::string mode, int epoch, int kfold); 
        void add_ntop_truth(std::string mode, std::string model, int epoch, int kfold, int data);
        void add_ntop_edge_accuracy(std::string mode, std::string model, int epoch, int kfold, int ntops, double data);
        void add_ntop_scores(std::string mode, std::string model, int epoch, int kfold, std::vector<double>* data);
        void add_ntru_ntop_scores(std::string mode, std::string model, int epoch, int kfold, int ntru, int ntop, double data);
        std::map<std::string, std::vector<cdata_t*>> get_plts(); 
        std::vector<std::string> model_names = {}; 
        std::vector<std::string> modes = {}; 
        std::vector<int> epochs = {}; 
        std::vector<int> kfolds = {}; 

        std::map<std::string, cmodel_t> model_data = {}; 
}; 

#endif
