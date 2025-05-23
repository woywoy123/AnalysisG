#ifndef METRICS_H
#define METRICS_H

#include <TH1F.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TMultiGraph.h>

#include <pyc/pyc.h>
#include <structs/report.h>
#include <structs/settings.h>
#include <templates/model_template.h>
#include <notification/notification.h>

#ifdef PYC_CUDA
#define cu_pyc c10::kCUDA
#else
#define cu_pyc c10::kCPU
#endif

struct analytics_t {
    model_template* model = nullptr; 
    model_report* report = nullptr; 

    int this_epoch = 0; 
    std::map<mode_enum, std::map<std::string, TH1F*>> loss_graph = {}; 
    std::map<mode_enum, std::map<std::string, TH1F*>> loss_node = {}; 
    std::map<mode_enum, std::map<std::string, TH1F*>> loss_edge = {}; 

    std::map<mode_enum, std::map<std::string, TH1F*>> accuracy_graph = {}; 
    std::map<mode_enum, std::map<std::string, TH1F*>> accuracy_node = {}; 
    std::map<mode_enum, std::map<std::string, TH1F*>> accuracy_edge = {}; 

    std::map<mode_enum, std::map<std::string, TH1F*>> pred_mass_edge = {}; 
    std::map<mode_enum, std::map<std::string, TH1F*>> truth_mass_edge = {}; 

    void purge(){
        this -> destroy(&this -> loss_graph);
        this -> destroy(&this -> loss_node);
        this -> destroy(&this -> loss_edge); 

        this -> destroy(&this -> accuracy_graph); 
        this -> destroy(&this -> accuracy_node);
        this -> destroy(&this -> accuracy_edge); 

        this -> destroy(&this -> pred_mass_edge); 
        this -> destroy(&this -> truth_mass_edge);
        delete this -> report; 
    }

    void destroy(std::map<mode_enum, std::map<std::string, TH1F*>>* data){
        std::map<mode_enum, std::map<std::string, TH1F*>>::iterator itr; 
        for (itr = data -> begin(); itr != data -> end(); ++itr){
            std::map<std::string, TH1F*>* mps = &itr -> second; 
            std::map<std::string, TH1F*>::iterator ith = mps -> begin();         
            for (; ith != mps -> end(); ++ith){delete ith -> second;}
        }
    }
}; 

class metrics: 
    public tools, 
    public notification
{
    public: 
        metrics(); 
        ~metrics(); 
      
        std::string output_path; 
        
        const std::vector<Color_t> colors_h = {
            kRed, kGreen, kBlue, kCyan, 
            kViolet, kOrange, kCoffee, kAurora
        }; 

        settings_t m_settings; 

        void dump_plots(int k); 
        void dump_loss_plots(int k); 
        void dump_accuracy_plots(int k); 
        void dump_mass_plots(int k); 

        model_report* register_model(model_template* model, int kfold); 
        void capture(mode_enum, int kfold, int epoch, int smpl_len); 

    private: 
        void build_th1f_loss(
                std::map<std::string, std::tuple<torch::Tensor*, lossfx*>>* type, 
                graph_enum g_num, int kfold
        ); 

        void add_th1f_loss(
                std::map<std::string, torch::Tensor>* type, 
                std::map<std::string, TH1F*>* lss_type,
                int kfold, int smpl_len
        ); 

        void build_th1f_accuracy(
                std::map<std::string, std::tuple<torch::Tensor*, lossfx*>>* type, 
                graph_enum g_num, int kfold
        ); 
        
        
        void add_th1f_accuracy(
                torch::Tensor* pred, torch::Tensor* truth, 
                TH1F* hist, int kfold, int smpl_len
        ); 


        void build_th1f_mass(std::string var_name, graph_enum typ, int kfold); 
        void add_th1f_mass(
                torch::Tensor* pmc, torch::Tensor* edge_index, 
                torch::Tensor* truth, torch::Tensor* pred, 
                int kfold, mode_enum mode, std::string var_name
        ); 


        void generic_painter(
                std::vector<TGraph*> k_graphs,
                std::string path, std::string title, 
                std::string xtitle, std::string ytitle, int epoch
        ); 

        std::map<std::string, std::vector<TGraph*>> build_graphs(
                std::map<std::string, TH1F*>* train,  std::map<std::string, float>* tr_, 
                std::map<std::string, TH1F*>* valid,  std::map<std::string, float>* va_, 
                std::map<std::string, TH1F*>* eval,   std::map<std::string, float>* ev_, 
                int ep
        ); 

        std::map<int, analytics_t> registry = {}; 

}; 

#endif
