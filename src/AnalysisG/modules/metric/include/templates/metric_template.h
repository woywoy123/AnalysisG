#ifndef METRIC_TEMPLATE_H
#define METRIC_TEMPLATE_H

#include <notification/notification.h>
#include <structs/property.h>
#include <structs/element.h>
#include <structs/event.h>
#include <structs/model.h>
#include <structs/report.h>
#include <meta/meta.h>

#include <tools/vector_cast.h>
#include <tools/merge_cast.h>
#include <tools/tools.h>

struct graph_t; 
class analysis; 
class model_template; 


class metric_template; 

struct metric_t {
    public: 
        int kfold = 0;
        int epoch = 0; 
        int device = 0; 
        
    private: 
        friend metric_template; 
        std::string* pth = nullptr;
        model_template* mdlx = nullptr; 
        std::map<graph_enum, std::vector<std::string>>* vars = nullptr; 
        std::map<graph_enum, std::vector<variable_t*>>* handl = nullptr; 
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
        void register_output(std::string name, T* t){ 
            if (this -> handle){return this -> handle -> process(&name) -> process(t, &name, nullptr);}
            if (!this -> _outdir.size()){this -> _outdir = this -> _name;}
            if (!this -> ends_with(&this -> _outdir, ".root")){this -> _outdir += ".root";}
            this -> handle = new write_t();
            this -> handle -> create(this -> _name, this -> _outdir);
            return this -> handle -> process(&name) -> process(t, &name, nullptr);
        }
  

        virtual void define_metric(); 

        cproperty<std::string, metric_template> name; 
        cproperty<std::map<std::string, std::string>, metric_template> run_names; 
        cproperty<std::map<std::string, std::string>, metric_template> variables; 
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

        void static set_variables(std::map<std::string, std::string>*, metric_template*); 
        void static get_variables(std::map<std::string, std::string>*, metric_template*);
        void static construct(
                std::map<graph_enum, std::vector<variable_t*>>* varx, 
                std::map<graph_enum, std::vector<std::string>>* req, 
                model_template* mdl, graph_t* grx
        );

        metric_template* clone(int);
        void link(std::string hsx, std::vector<graph_t*>* data, mode_enum mx); 
        bool link(model_template*);

        void execute(metric_t* mtx); 

        void define(); 

        std::map<int, torch::TensorOptions*> get_devices(); 
        std::vector<int> get_kfolds(); 

        write_t* handle = nullptr; 
}; 


#endif
