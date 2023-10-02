#include "../abstractions/abstractions.h"
#include "../abstractions/cytypes.h"

#ifndef PLOTTING_H
#define PLOTTING_H

#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>


struct paint_t
{
    float n_events = 0; 

    bool atlas_style = false; 
    bool root_style = false; 
    bool mpl_style = true; 
    bool autoscale = true;
    bool latex = true;

    std::string color = ""; 
    std::vector<std::string> colors = {}; 

    std::string marker = "."; 
    std::vector<std::string> markers = {
        ",", ".", "--", "x", "o", "O"
    }; 

    std::string texture = "";
    std::vector<std::string> textures = {
        "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"
    }; 

    float xscaling = 1.25*6.4; 
    float yscaling = 1.25*4.8;
    float alpha = 0.5; 
    float line_width = 1; 

    float font_size = 10; 
    float label_size = 12.5; 
    float title_size = 10; 
    float legend_size = 10;
    std::string legend_loc = "upper left"; 
    std::string hist_fill = "fill"; 

    int atlas_loc = 2; 
    float atlas_lumi = -1; 
    float atlas_com = -1;
    float atlas_year = -1; 
    bool atlas_data = false; 
};

struct axis_t
{
    std::string title = ""; 
    std::string dim = ""; 
    std::map<std::string, std::vector<float>> sorted_data = {}; 
    std::map<std::string, float> label_data = {}; 
    std::vector<float> random_data     = {};
    std::vector<float> random_data_up  = {}; 
    std::vector<float> random_data_down = {}; 

    int bins = 0; 
    float start = -1; 
    float end = -1; 
    float step = 0; 
    bool underflow = false; 
    bool overflow = false;
    bool bin_centering = false;  
    bool logarithmic = false; 
    
}; 

struct io_t
{
    std::string filename = "untitled"; 
    std::string outputdir = "./Plots";
    int dpi = 250; 
}; 

struct figure_t
{
    std::string title = "no-title";
    bool label_data = false; 
    bool overlay = false; 
    bool histogram = false; 
    bool line = false; 
}; 

struct abstract_plot
{
    paint_t  cosmetic; 
    figure_t figure; 
    io_t     file; 
    axis_t   x; 
    axis_t   y; 
    std::map<std::string, axis_t> stacked = {}; 
}; 




class CyPlotting
{
    public:
        CyPlotting(); 
        ~CyPlotting(); 

        paint_t  painter_params;  
        io_t     file_params; 
        figure_t figure_params; 
}; 


struct node_t
{
    std::map<std::string, int> nodes = {};  
    bool called = false; 
    void init(metric_t* met){ 
        if (called){return;}
        nodes = met -> num_nodes; 
        called = true; 
    }
    void collect(axis_t* ax){
        std::map<std::string, int>::iterator itr = nodes.begin();  
        for (; itr != nodes.end(); ++itr){
            (*ax).label_data[itr -> first] = itr -> second; 
        }
    }
}; 

struct roc_t
{
    std::map<int, float> auc = {}; 
    std::map<int, std::vector<float>> tpr = {}; 
    std::map<int, std::vector<float>> fpr = {}; 
    std::map<int, std::vector<float>> threshold = {}; 
    std::vector<std::vector<float>> truth = {}; 
    std::vector<std::vector<float>> pred = {}; 
    std::vector<std::vector<int>> confusion = {};
}; 

struct stats_t
{
    std::string variable = "";  
    bool classifier = false;
    bool regression = false; 
    float average = 0; 
    float up = 0; 
    float down = 0; 
    float stdev = 0; 
}; 

struct atomic_t
{
    std::map<std::string, stats_t> loss = {}; 
    std::map<std::string, stats_t> accuracy = {};
    std::map<std::string, roc_t>   roc_curve = {}; 
}; 


struct epoch_t
{
    int epoch = -1; 
    std::map<int, std::map<std::string, atomic_t>> data = {}; 
    std::map<int, std::map<std::string, node_t>> nodes = {};
    void initialize(int e){ epoch = e; }


    std::string get_fold(std::string kfold){ 
        std::vector<std::string> spl = Tools::split(kfold, "-"); 
        if (!spl.size()){return "None";}
        return spl[spl.size()-1]; 
    }

    int get_mode(std::string mode){
        if (mode == "training"){ return 0; }
        if (mode == "validation"){ return 1; }
        if (mode == "evaluation"){ return 2; }
        return 3; 
    }

    template <typename G>
    std::tuple<bool, G> get_this(
            std::string mode, 
            std::string kfold, 
            std::map<int, std::map<std::string, G>>* get)
    {
        int mo = get_mode(mode);
        if (get -> count(mo)){}
        else {return std::make_tuple(false, G());}

        std::string fold = get_fold(kfold); 
        if ((*get)[mo].count(fold)){}
        else {return std::make_tuple(false, G());}

        return std::make_tuple(true, (*get)[mo][fold]); 
    }
    
    std::tuple<bool, node_t> get_nodes(std::string mode, std::string kfold){
        return get_this(mode, kfold, &nodes); 
    }

    std::tuple<bool, atomic_t> get_atomic(std::string mode, std::string kfold){
        return get_this(mode, kfold, &data); 
    }

    void build_variables(
            std::map<std::string, stats_t>* stat,
            std::map<std::string, float>* avg, 
            std::map<std::string, std::vector<std::vector<float>>>* preds)
    {
        std::vector<std::string> spl; 
        std::map<std::string, float>::iterator itf = avg -> begin(); 
        for (; itf != avg -> end(); ++itf){
            spl = Tools::split(itf -> first, "_");
            if (spl.size() == 0){continue;}

            std::string var = Tools::join(&spl, 0, spl.size() -1, "_"); 
            if (stat -> count(var)){continue;}
            (*stat)[var].average = (*avg)[var + "_average"]; 
            (*stat)[var].stdev   = (*avg)[var + "_stdev"]; 
            (*stat)[var].down    = (*avg)[var + "_down"]; 
            (*stat)[var].up      = (*avg)[var + "_up"]; 
            (*stat)[var].classifier = (*preds)[var][0].size() > 1;
            (*stat)[var].regression = (*preds)[var][0].size() == 1;
            (*stat)[var].variable = var; 
        }
    }

    std::vector<roc_t*> release_roc()
    {
        std::vector<roc_t*> output = {}; 
        std::map<std::string, atomic_t>::iterator itr; 
        std::map<std::string, roc_t>::iterator itc; 
        for (unsigned int x = 0; x != data.size(); ++x){
            if (!data.count(x)){continue;}
            for (itr = data[x].begin(); itr != data[x].end(); ++itr){
                std::string name = itr -> first; 
                itc = data[x][name].roc_curve.begin();
                for (; itc != data[x][name].roc_curve.end(); ++itc){
                    if (!itc -> second.truth.size()){continue;}
                    output.push_back(&(itc -> second)); 
                }
            }
        }
        return output; 
    }



    void prepare_roc(std::string mode, std::string kfold, metric_t* met){
        atomic_t* atm = &(data[get_mode(mode)][get_fold(kfold)]); 
        std::map<std::string, stats_t>::iterator itr = atm -> loss.begin(); 
        for (; itr != atm -> loss.end(); ++itr){
            if (!itr -> second.classifier){continue;}
            std::string name = itr -> first; 
            atm -> roc_curve[name].truth = met -> truth[name]; 
            atm -> roc_curve[name].pred  = met -> pred[name]; 
        }
    }

    void append_accuracy(std::string mode, std::string kfold, metric_t* met){
        std::string fold = get_fold(kfold); 
        int mode_i = get_mode(mode); 
        build_variables(&data[mode_i][fold].accuracy, &met -> acc_average, &met -> pred); 
    }

    void append_loss(std::string mode, std::string kfold, metric_t* met){
        std::string fold = get_fold(kfold); 
        int mode_i = get_mode(mode); 
        build_variables(&data[mode_i][fold].loss, &met -> loss_average, &met -> pred); 
    }

    void append_nodes(std::string mode, std::string kfold, metric_t* met){
        std::string fold = get_fold(kfold);
        int mode_i = get_mode(mode); 
        nodes[mode_i][fold].init(met);
    }

    void add(std::string mode, std::string kfold, metric_t* met){
        append_nodes(mode, kfold, met); 
        append_loss(mode, kfold, met); 
        append_accuracy(mode, kfold, met); 
        prepare_roc(mode, kfold, met); 
    }
}; 


struct report_t
{
    int current_epoch;
    std::map<std::string, float> auc_train;
    std::map<std::string, float> auc_valid; 
    std::map<std::string, float> auc_eval; 

    std::map<std::string, float> loss_train;
    std::map<std::string, float> loss_valid; 
    std::map<std::string, float> loss_eval; 

    std::map<std::string, float> loss_train_up;
    std::map<std::string, float> loss_valid_up; 
    std::map<std::string, float> loss_eval_up; 

    std::map<std::string, float> loss_train_down;
    std::map<std::string, float> loss_valid_down; 
    std::map<std::string, float> loss_eval_down; 

    std::map<std::string, float> acc_train;
    std::map<std::string, float> acc_valid; 
    std::map<std::string, float> acc_eval; 

    std::map<std::string, float> acc_train_up;
    std::map<std::string, float> acc_valid_up; 
    std::map<std::string, float> acc_eval_up; 

    std::map<std::string, float> acc_train_down;
    std::map<std::string, float> acc_valid_down; 
    std::map<std::string, float> acc_eval_down; 
};




class CyMetric
{
    public:
        CyMetric(); 
        ~CyMetric();
        std::vector<roc_t*> FetchROC(); 

        void dress_abstraction(
                abstract_plot* inpt,
                std::string title, std::string xtitle, std::string ytitle, 
                std::string fname, bool hist, bool line, int n_events);

        void AddMetric(std::map<std::string, metric_t>* values, std::string mode);  
        void BuildPlots(std::map<std::string, abstract_plot>* output); 

        void BuildAccuracy(std::vector<abstract_plot>* plt, std::vector<int> these);
        void BuildLoss(std::vector<abstract_plot>* plt, std::vector<int> these);
        void BuildROC(std::vector<abstract_plot>* plt, std::vector<int> these); 
        void BuildNodes(abstract_plot* plt, epoch_t* ep); 



        std::string outpath; 
        report_t report; 
        int current_epoch = -1; 
        std::map<int, epoch_t> epoch_data = {};
        
}; 


#endif 
