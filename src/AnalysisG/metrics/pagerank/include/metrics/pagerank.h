#ifndef PAGERANK_METRIC_H
#define PAGERANK_METRIC_H

#include <templates/metric_template.h>

class pagerank_metric: public metric_template
{
    public:
        pagerank_metric(); 
        ~pagerank_metric() override; 
        pagerank_metric* clone() override; 

        void define_metric(metric_t* mtx) override; 
        void define_variables() override; 
        void event() override; 
        void batch() override; 
        void end() override; 

        void pagerank(
            std::map<int, std::map<std::string, particle_template*>>* clust, 
            std::map<std::string, std::vector<particle_template*>>* out,
            std::map<std::string, float>* bin_out,
            std::map<int, std::map<int, float>>* bin_data,
            int batch_offset
        ); 

        std::vector<particle_template*> build_top(std::map<int, std::map<int, particle_template*>>*); 

        std::map<std::string, std::map<std::string, long>> file_maps; 
        std::map<std::string, std::map<std::string, long>> file_stats;  

        float  alpha    = 0.85; 
        float  norm_lim = 1e-6; 
        size_t max_itr  = 1e6; 
        int kfold = -1; 
        int epoch = -1; 

    private: 
        std::string mode = ""; 

        int process_mapping = -1; 
        std::vector<int>    top_truth_num_nodes; 
        std::vector<double> top_truth_pt; 
        std::vector<double> top_truth_eta;
        std::vector<double> top_truth_phi; 
        std::vector<double> top_truth_energy; 

        std::vector<double> top_truth_px; 
        std::vector<double> top_truth_py;
        std::vector<double> top_truth_pz; 
        std::vector<double> top_truth_mass;


        std::vector<int>    top_pr_reco_num_nodes; 
        std::vector<double> top_pr_reco_pt; 
        std::vector<double> top_pr_reco_eta;
        std::vector<double> top_pr_reco_phi; 
        std::vector<double> top_pr_reco_energy; 

        std::vector<double> top_pr_reco_px; 
        std::vector<double> top_pr_reco_py;
        std::vector<double> top_pr_reco_pz; 
        std::vector<double> top_pr_reco_mass;
        std::vector<float>  top_pr_reco_pagerank; 


        std::vector<int>    top_nom_reco_num_nodes; 
        std::vector<double> top_nom_reco_pt; 
        std::vector<double> top_nom_reco_eta;
        std::vector<double> top_nom_reco_phi; 
        std::vector<double> top_nom_reco_energy; 

        std::vector<double> top_nom_reco_px; 
        std::vector<double> top_nom_reco_py;
        std::vector<double> top_nom_reco_pz; 
        std::vector<double> top_nom_reco_mass;
        std::vector<float>  top_nom_reco_score; 

}; 

struct kinematic_t {
    double px = 0; 
    double py = 0; 
    double pz = 0; 
    double mass = 0; 
    
    double pt  = 0;     
    double eta = 0; 
    double phi = 0; 
    double energy = 0; 

    int num_nodes = 0; 
    double score = 0; 
}; 

struct meta_event_t {
    int process_mapping = -1; 
    std::vector<kinematic_t> truth = {}; 
    std::vector<kinematic_t> pageranked = {}; 
    std::vector<kinematic_t> nominal = {}; 
}; 



struct sample_t {
    std::string meta_data = ""; 
    std::map<std::string, std::map<std::string, long>> file_map  = {}; 
    std::map<std::string, std::map<std::string, long>> file_stat = {}; 
    std::map<std::string, std::map<long, meta_event_t>> data = {}; 
}; 


class collector: public tools
{
    public:
        collector();
        ~collector(); 
        void set_index(long idx_); 
        
        kinematic_t create_kinematic(
                double px, double py,  double pz,  double mass,
                double pt, double eta, double phi, double energy,  
                int num_nodes, double score
        ); 

        void add_truth(kinematic_t* p, std::string mode, std::string model, int kfold, int epoch);
        void add_pagerank(kinematic_t* p, std::string mode, std::string model, int kfold, int epoch);
        void add_nominal(kinematic_t* p, std::string mode, std::string model, int kfold, int epoch);
        void add_process(int prc, std::string mode, std::string model, int kfold, int epoch);
        bool add_meta(std::string data, std::string model, int kfold, int epoch);
        void add_file_map(std::string fname, long idx_, std::string mode, std::string model, int kfold, int epoch, bool stat); 

    private:
        long  idx = 0; 
        std::map<std::string, std::map<int, std::map<int, sample_t>>> data;
}; 

#endif
