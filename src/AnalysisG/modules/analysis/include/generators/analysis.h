#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <generators/sampletracer.h>
#include <generators/dataloader.h>
#include <generators/optimizer.h>

#include <templates/graph_template.h>
#include <templates/event_template.h>
#include <templates/selection_template.h>
#include <templates/model_template.h>
#include <templates/fx_enums.h>

#include <structs/settings.h>
#include <io/io.h>

class analysis: 
    public notification, 
    public tools
{
    public: 
        analysis();
        ~analysis(); 

        void add_samples(std::string path, std::string label);

        void add_selection_template(selection_template* sel); 
        void add_event_template(event_template* ev, std::string label); 
        void add_graph_template(graph_template* gr, std::string label); 

        void add_model(model_template* model, optimizer_params_t* op, std::string run_name); 
        void start(); 

        std::map<std::string, float> progress(); 
        std::map<std::string, std::string> progress_mode(); 
        std::map<std::string, std::string> progress_report(); 
        std::map<std::string, bool> is_complete();

        void attach_threads(); 
        settings_t m_settings; 

    private:
        void build_project(); 
        void build_events(); 
        void build_selections(); 
        void build_graphs(); 
        void build_dataloader(); 
        void build_model_session(); 
       
        std::map<std::string, std::string> file_labels = {}; 
        std::map<std::string, event_template*> event_labels = {}; 
        std::map<std::string, selection_template*> selection_names = {}; 

        std::map<std::string, std::map<std::string, graph_template*>> graph_labels = {}; 
        std::vector<std::tuple<model_template*, optimizer_params_t*>> model_sessions = {}; 
        std::vector<std::string> model_session_names = {}; 

        std::map<std::string, optimizer*> trainer = {};
        std::map<std::string, model_report*> reports = {}; 
        std::vector<std::thread*> threads = {}; 

        dataloader*   loader = nullptr; 
        sampletracer* tracer = nullptr; 
        io*           reader = nullptr; 

}; 

#endif