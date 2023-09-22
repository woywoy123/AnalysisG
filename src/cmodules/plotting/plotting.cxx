#include "../plotting/plotting.h"

CyPlotting::CyPlotting(){}
CyPlotting::~CyPlotting(){}

CyMetric::CyMetric(){}
CyMetric::~CyMetric(){}

void CyMetric::AddMetric(std::map<std::string, metric_t>* values, std::string mode)
{
    int c_epoch = this -> current_epoch; 
    if (this -> epoch_data.count(c_epoch)){}
    else { this -> epoch_data[c_epoch].initialize(c_epoch); }

    epoch_t* this_epoch = &(this -> epoch_data[c_epoch]);
    std::map<std::string, metric_t>::iterator k_itr = values -> begin(); 
    for (; k_itr != values -> end(); ++k_itr){
        this_epoch -> add(mode, k_itr -> first, &k_itr -> second); 
    }
}

std::vector<roc_t*> CyMetric::FetchROC()
{
    epoch_t* epoch = &(this -> epoch_data[this -> current_epoch]); 
    return epoch -> release_roc();
}

void CyMetric::BuildPlots(std::map<std::string, abstract_plot>* output)
{
    auto dress_abstraction = [this](
            abstract_plot* inpt, 
            std::string title, std::string xtitle, std::string ytitle,
            std::string fname,  bool hist, bool line, int n_events)
    {
        inpt -> cosmetic.n_events = n_events; 
        inpt -> cosmetic.atlas_style = true; 
        inpt -> cosmetic.atlas_year = 2016;
        inpt -> cosmetic.atlas_com = 13; 
        inpt -> figure.title = title; 
        inpt -> figure.histogram = hist; 
        inpt -> figure.line = line; 
        inpt -> file.filename = fname;
        inpt -> file.outputdir = this -> outpath; 
        inpt -> x.title = xtitle; 
        inpt -> y.title = ytitle; 
    }; 




    std::map<std::string, abstract_plot> tmp = {}; 
    std::map<int, epoch_t>* ep = &(this -> epoch_data); 

    std::vector<int> these_eps = {}; 
    for (int x = 0; x < ep -> size(); ++x){
        if (!ep -> count(x)){continue;}
        these_eps.push_back(x); 
    }
    if (!these_eps.size()){return;}
    
    // Get the node data first
    abstract_plot nodes; 
    dress_abstraction(
            &nodes, "Node Distributions of Sample", 
            "Number of Nodes", "Number of Entries", 
            "NodeStatistics", true, false, -1); 
    
   ep -> at(this -> current_epoch).draw_nodes(&nodes); 








}
