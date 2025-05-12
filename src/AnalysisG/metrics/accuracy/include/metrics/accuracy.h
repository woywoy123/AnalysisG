#ifndef AVERAGE_METRIC_H
#define AVERAGE_METRIC_H

#include <templates/metric_template.h>

class accuracy_metric: public metric_template
{
    public:
        accuracy_metric(); 
        ~accuracy_metric() override; 
        accuracy_metric* clone() override; 

        void define_metric(metric_t* mtx) override; 
        void define_variables() override; 
        void event() override; 
        void batch() override; 
        void end() override; 

    private: 
        std::string mode = ""; 
        long idx = 0; 
        float edge_accuracy = 0;
        float global_edge_accuracy = 0; 

        int ntop_truth = 0; 
        std::vector<float> ntop_scores = {}; 

        std::vector<float> ntops_accuracy = {}; 
        std::map<int, int> _global_edge_accuracy = {}; 
        std::map<int, std::map<int, int>> ntop_accuracy = {}; 
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

        cdata_t* get_mode(
            std::string model, std::string mode, 
            int epoch, int kfold
        ); 

        void add_ntop_truth(
            std::string mode, std::string model, 
            int epoch, int kfold, int data
        );

        void add_ntop_edge_accuracy(
            std::string mode, std::string model, 
            int epoch, int kfold, int ntops, double data
        );

        void add_ntop_scores(
            std::string mode, std::string model, 
            int epoch, int kfold, std::vector<double>* data
        );

        void add_ntru_ntop_scores(
            std::string mode, std::string model, 
            int epoch, int kfold, int ntru, int ntop, double data
        );

        std::map<std::string, std::vector<cdata_t*>> get_plts(); 
        
        std::vector<std::string> model_names = {}; 
        std::vector<std::string> modes = {}; 
        std::vector<int> epochs = {}; 
        std::vector<int> kfolds = {}; 

        std::map<std::string, cmodel_t> model_data = {}; 
}; 

#endif
