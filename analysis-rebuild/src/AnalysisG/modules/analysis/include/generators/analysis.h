#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <io/io.h>
#include <generators/sampletracer.h>
#include <generators/dataloader.h>
#include <generators/optimizer.h>

#include <templates/model_template.h>
#include <templates/fx_enums.h>

class analysis: 
    public notification, 
    public tools
{
    public: 
        analysis();
        ~analysis(); 

        void add_samples(std::string path, std::string label); 
        void add_event_template(event_template* ev, std::string label); 
        void add_graph_template(graph_template* gr, std::string label); 
        void add_model(model_template* model, optimizer_params_t* op, std::string run_name); 
        void start(); 

        // settings
        std::string output_path = "./ProjectName"; 

        // optimizer 
        int epochs = 10; 
        int kfolds = 10; 
        int num_examples = 3; 
        float train_size = 50; 
        
        bool training = true;
        bool validation = true; 
        bool evaluation = true;
        bool continue_training = false; 

        std::string var_pt = "pt"; 
        std::string var_eta = "eta";
        std::string var_phi = "phi";
        std::string var_energy = "energy"; 
        std::vector<std::string> targets = {"top_edge"}; 

        int nbins = 400; 
        int refresh = 10; 
        int max_range = 400; 

    private:
        void build_project(); 
        void build_events(); 
        void build_graphs(); 
        void build_dataloader(); 
        void build_model_session(); 
       
        std::map<std::string, std::string> file_labels = {}; 
        std::map<std::string, event_template*> event_labels = {}; 
        std::map<std::string, std::map<std::string, graph_template*>> graph_labels = {}; 
        std::vector<std::tuple<model_template*, optimizer_params_t*>> model_sessions = {}; 
        std::vector<std::string> model_session_names = {}; 

        optimizer*   trainer = nullptr; 
        dataloader*   loader = nullptr; 
        sampletracer* tracer = nullptr; 
        io*           reader = nullptr; 

}; 

#endif
