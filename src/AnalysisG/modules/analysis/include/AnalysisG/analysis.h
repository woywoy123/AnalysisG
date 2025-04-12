#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <string>
#include <generators/sampletracer.h>
#include <generators/dataloader.h>
#include <generators/optimizer.h>

#include <templates/graph_template.h>
#include <templates/event_template.h>
#include <templates/metric_template.h>
#include <templates/selection_template.h>
#include <templates/model_template.h>
#include <structs/settings.h>
#include <io/io.h>

template <typename g>
void flush(std::map<std::string, g*>* data){
    typename std::map<std::string, g*>::iterator tx = data -> begin(); 
    for (; tx != data -> end(); ++tx){delete tx -> second;}
    data -> clear(); 
}

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
        void add_metric_template(metric_template* mx, model_template* mdl);

        void add_model(model_template* model, optimizer_params_t* op, std::string run_name); 
        void add_model(model_template* model, std::string run_name); 
        void attach_threads(); 
        void start(); 

        std::map<std::string, std::vector<float>> progress(); 
        std::map<std::string, std::string> progress_mode(); 
        std::map<std::string, std::string> progress_report(); 
        std::map<std::string, bool> is_complete();

        settings_t m_settings; 
        std::map<std::string, meta*> meta_data = {}; 

    private:

        void check_cache(); 
        void build_project(); 
        void build_events(); 
        void build_selections(); 
        void build_graphs(); 
        void build_model_session(); 
        void build_inference();
        bool build_metric(); 
        void build_metric_folds();

        void build_dataloader(bool training); 
        void fetchtags(); 
        bool started = false;  

        static int add_content(
            std::map<std::string, torch::Tensor*>* data, 
            std::vector<variable_t>* content, int index, 
            std::string prefx, TTree* tt = nullptr
        ); 

        static void add_content(
            std::map<std::string, torch::Tensor*>* data, std::vector<std::vector<torch::Tensor>>* buff, 
            torch::Tensor* edge, torch::Tensor* node, torch::Tensor* batch, std::vector<long> mask
        ); 

        static void execution(
            model_template* mdx, model_settings_t mds, std::vector<graph_t*>* data, size_t* prg,
            std::string output, std::vector<variable_t>* content, std::string* msg
        );

        static void initialize_loop(
            optimizer* op, int k, model_template* model, 
            optimizer_params_t* config, model_report** rep
        );

        template <typename g>
        void safe_clone(std::map<std::string, g*>* mp, g* in){
            std::string name = in -> name; 
            if (mp -> count(name)){return;}
            (*mp)[name] = in -> clone(1); 
        }

        std::map<std::string, std::string> file_labels = {}; 
        std::map<std::string, event_template*> event_labels = {}; 
        std::map<std::string, metric_template*> metric_names = {}; 
        std::map<std::string, selection_template*> selection_names = {}; 
        std::map<std::string, std::map<std::string, graph_template*>> graph_labels = {}; 

        std::vector<std::string> model_session_names = {}; 
        std::map<std::string, model_template*> model_inference = {}; 
        std::map<std::string, model_template*> model_metrics   = {}; 
        std::vector<std::tuple<model_template*, optimizer_params_t*>> model_sessions = {}; 

        std::map<std::string, optimizer*   > trainer = {};
        std::map<std::string, model_report*> reports = {}; 
        std::vector<std::thread*> threads = {};

        std::map<std::string, std::map<std::string, bool>> in_cache = {}; 
        std::map<std::string, bool> skip_event_build = {}; 
        std::map<std::string, std::string> graph_types = {}; 

        std::vector<folds_t>* tags  = nullptr; 
        dataloader*          loader = nullptr; 
        sampletracer*        tracer = nullptr; 
        io*                  reader = nullptr; 

}; 

#endif
