#ifndef METRIC_TEMPLATE_H
#define METRIC_TEMPLATE_H

#include <notification/notification.h>
#include <structs/property.h>
#include <structs/element.h>
#include <structs/event.h>
#include <structs/model.h>
#include <structs/enums.h>
#include <meta/meta.h>

#include <plotting/plotting.h>
#include <tools/vector_cast.h>
#include <tools/merge_cast.h>
#include <tools/tools.h>

struct graph_t; 
class analysis; 
class model_template;
class metric_template; 

struct metric_t {
    public: 
        ~metric_t(); 

        int kfold = 0;
        int epoch = 0; 
        int device = 0; 

        template <typename g>
        g get(graph_enum grx, std::string _name){
            g out = g(); 
            if (!this -> h_maps[grx][_name]){
                std::cout << "\033[1;31m Variable not found: " << _name << "\033[0m" << std::endl;
                return out;
            }
            size_t idx = this -> v_maps[grx][_name]; 
            variable_t* v = (*this -> handl)[grx][idx]; 
            if (!v){
                std::cout << "\033[1;31m Variable not found: " << _name << "\033[0m" << std::endl;
                return out;
            }
            if (v -> element(&out)){return out;}
            std::cout << "\033[1;31m Expected Type: " << v -> as_string(); 
            std::cout << " -> " << _name << "\033[0m" << std::endl;
            return out; 
        }
        std::string mode(); 

    private: 
        friend metric_template; 
        friend analysis; 

        void build(); 
        mode_enum train_mode; 
        std::string* pth = nullptr;
        model_template* mdlx = nullptr;
        metric_template* mtx = nullptr;
        size_t index = 0; 
         
        std::map<graph_enum, std::vector<std::string>>* vars = nullptr; 
        std::map<graph_enum, std::vector<variable_t*>>* handl = nullptr; 
        std::map<graph_enum, std::map<std::string, size_t>> v_maps = {}; 
        std::map<graph_enum, std::map<std::string, bool>>   h_maps = {}; 
}; 






class metric_template: 
    public tools, 
    public notification
{
    public:
        metric_template(); 
        virtual ~metric_template(); 
        virtual metric_template* clone(); 

        template <typename T>
        void register_output(std::string tree, std::string __name, T* t){ 
            if (this -> handle){return this -> handle -> process(&tree, &__name, t);}
            this -> handle = new writer();
            this -> handle -> create(&this -> _outdir); 
            this -> handle -> process(&tree, &__name, t); 
        }

        template <typename T>
        void write(std::string tree, std::string __name, T* t, bool fill = false){
            if (!this -> handle){return;}
            this -> handle -> process(&tree, &__name, t); 
            if (!fill){return;}
            this -> handle -> write(&tree);
        }

        virtual void define_variables(); 
        virtual void define_metric(metric_t* v); 
        virtual void event();
        virtual void batch();
        virtual void end(); 

        cproperty<std::string, metric_template> name; 
        cproperty<std::map<std::string, std::string>, metric_template> run_names; 
        cproperty<std::vector<std::string>, metric_template> variables; 
        meta* meta_data = nullptr; 

    private:
        friend analysis;

        std::map<std::string, model_template*> lnks; 
        std::map<std::string, std::vector<model_template*>> hash_mdl = {}; 
        std::map<std::string, std::map<mode_enum, std::vector<graph_t*>*>> hash_bta = {}; 
        std::map<std::string, std::map<int, std::map<int, std::string>>> _epoch_kfold;
        std::map<std::string, std::map<graph_enum, std::vector<std::string>>> _var_type; 
        
        std::string _name = "metric-template"; 
        std::string _outdir = ""; 

        std::map<std::string, std::string> _run_names = {}; 
        std::map<std::string, std::string> _variables = {}; 

        void static set_name(std::string*, metric_template*); 
        void static get_name(std::string*, metric_template*);

        void static set_run_name(std::map<std::string, std::string>*, metric_template*); 
        void static get_run_name(std::map<std::string, std::string>*, metric_template*);

        void static set_variables(std::vector<std::string>*, metric_template*); 
        void static get_variables(std::vector<std::string>*, metric_template*);
        void static construct(
                std::map<graph_enum, std::vector<variable_t*>>* varx, 
                std::map<graph_enum, std::vector<std::string>>* req, 
                model_template* mdl, graph_t* grx, std::string* mtx
        );

        metric_template* clone(int);
        bool link(model_template*);
        void link(std::string hsx, std::vector<graph_t*>* data, mode_enum mx); 
        void execute(metric_t* mtx, metric_template* obj, size_t* prg, std::string* msg); 
        void define(
                std::vector<metric_t*>* vr, std::vector<size_t>* num, 
                std::vector<std::string*>* title, size_t* offset
        ); 

        size_t size(); 

        std::map<int, torch::TensorOptions*> get_devices(); 
        std::vector<int> get_kfolds(); 

        writer* handle = nullptr; 
}; 


#endif
