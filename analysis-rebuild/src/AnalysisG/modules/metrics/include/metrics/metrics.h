#ifndef METRICS_H
#define METRICS_H

#include <TH1F.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TMultiGraph.h>

#include <templates/model_template.h>

enum mode_enum {training, validation, evaluation}; 

struct analytics_t {
    model_template* model = nullptr; 

    int this_epoch = 0; 
    std::map<mode_enum, std::map<std::string, TH1F*>> loss_graph = {}; 
    std::map<mode_enum, std::map<std::string, TH1F*>> loss_node = {}; 
    std::map<mode_enum, std::map<std::string, TH1F*>> loss_edge = {}; 

    std::map<mode_enum, std::map<std::string, TH1F*>> pred_mass_edge = {}; 
    std::map<mode_enum, std::map<std::string, TH1F*>> truth_mass_edge = {}; 
}; 

class metrics: public tools
{
    public: 
        metrics(); 
        ~metrics(); 
      
        Int_t epochs; 

        std::string output_path; 
        
        const std::vector<Color_t> colors_h = {
            kRed, kGreen, kBlue, kCyan, kViolet, kOrange, kCoffee, kAurora
        }; 


        void dump_plots(); 
        void dump_loss_plots(); 
        void dump_mass_plots(); 

        void register_model(model_template* model, int kfold); 
        void capture(mode_enum, int kfold, int epoch, int smpl_len); 

    private: 
        void build_th1f_loss(
                std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>* type, 
                graph_enum g_num, int kfold
        ); 

        void build_th1f_mass(std::string var_name, graph_enum typ, int kfold); 

        void add_th1f_loss(
                std::map<std::string, torch::Tensor>* type, 
                std::map<std::string, TH1F*>* lss_type,
                int kfold, int smpl_len
        ); 

        void add_th1f_mass(
                std::map<std::string, torch::Tensor*>* node_feats, 
                torch::Tensor* edge_index, 
                torch::Tensor* truth, 
                torch::Tensor* pred, 
                int kfold, mode_enum mode
        ); 


        void generic_painter(
                std::vector<TGraph*> k_graphs,
                std::string path, std::string title, 
                std::string xtitle, std::string ytitle
        ); 

        std::map<std::string, std::vector<TGraph*>> build_graphs(
                std::map<std::string, TH1F*>* train, 
                std::map<std::string, TH1F*>* valid, 
                std::map<std::string, TH1F*>* eval
        ); 

        std::map<int, analytics_t> registry = {}; 

}; 

#endif
