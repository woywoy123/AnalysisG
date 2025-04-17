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

//! analysis class
/*! 
    The main interface class used to combine all submodules into a single object.
    From this interface, the runtime is defined for the various templates; event_template, selection_template, graph_template and metric_template.
*/

class analysis: 
    public notification, 
    public tools
{
    public: 
        //! constructor of the analysis
        analysis();

        //! destructor of the analysis
        ~analysis(); 

        //! Specifies the samples that should be processed, along with any labels.
        /*! 
            The the path variable can have the relative path ./, ./*.root or absolute /some/path/to/*.root syntax.
            If the given sample requires some additional labelling, i.e. these samples need to be processed using a special 
            event_template or graph_template definition, then ROOT samples with those labels will be processed.
        */
        void add_samples(std::string path, std::string label);

        /*!
            Registers the given selection_template for further processing.
            After the framework has finished processing, the selection_template will be merged and can be used to expose 
            any generated C++ objects into python. 
        */
        void add_selection_template(selection_template* sel); 


        /*! 
            Registers the given event_template and processes ROOT samples specified by the add_samples function with the given label.
        */
        void add_event_template(event_template* ev, std::string label); 

        /*! 
            Registers the given graph_template and processes ROOT samples specified by the add_samples function with the given label.
            To generate graphs, a compatible event_template needs to be implemented. Once these have been processed, they can be cached 
            and fetched without specifying the event_template. This speeds up processing times.
        */
        void add_graph_template(graph_template* gr, std::string label); 

        /*! 
            A metric that should be used for post analysis of a trained GNN or other MVA.
            This is a relatively new feature and is mostly experimental.
        */
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

        /*! 
            Checks the local graph_template cache.
            If anything has been found, then it will decompress them from the HDF5 files.
            For special cases, where multiple graph_templates have been cached, specifiying a graph_template implementation in the add_graph_template will cause only those to be fetched.
        */
        void check_cache(); 

        /*! 
            Builds the initial directories.
        */
        void build_project(); 

        /*!
            Compiles the ROOT samples into event_template objects that can be further processed into selection_templates or graph_templates.
        */
        void build_events(); 

        /*!
            Compiles the event_templates into selection_templates and performs additional selections.
        */
        void build_selections(); 

        /*!
            Compiles the event_templates into graph_templates which can be either cached (graph_t) and used for training a GNN or evaluating it (see metric_template or add_model).
        */
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

        static void execution_metric(metric_t* mt, size_t* prg, std::string* msg); 

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
