#ifndef METRIC_TEMPLATE_H
#define METRIC_TEMPLATE_H

#include <templates/particle_template.h>
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
struct metric_t; 
class analysis; 
class model_template;
class metric_template; 

template <typename T> 
struct router_t {
    router_t(std::string tree, std::string leaf){
        this -> tr_name = tree; 
        this -> lf_name = leaf; 
    } 

    T training; 
    T validation; 
    T evaluation; 
    T* get(mode_enum mode){
        switch(mode){
            case mode_enum::training:   return &this -> training; 
            case mode_enum::validation: return &this -> validation;
            case mode_enum::evaluation: return &this -> evaluation; 
            default: return nullptr; 
        }
        return nullptr; 
    }
    
    void write(metric_template* mtx, metric_t* mti, T* inpt, bool fill = false); 
    std::string lf_name = ""; 
    std::string tr_name = ""; 

}; 

struct metric_t : 
    public tools,
    public notification 
{
    public: 
        metric_t(); 
        ~metric_t(); 
    
        int kfold = 0;
        int epoch = 0; 
        int device = 0; 
        long index = 0; 

        std::string mode(); 
        std::string* get_filename(long unsigned int idx); 

        template <typename g>
        g get(graph_enum grx, std::string _name){
            g out = g(); 
            if (!this -> h_maps[grx][_name]){
                this -> coms -> message(this -> emsg + _name); return out;
            }
            size_t idx = this -> v_maps[grx][_name]; 
            variable_t* v = (*this -> handl)[grx][idx]; 
            if (!v){this -> coms -> message(this -> emsg + _name ); return out;}

            if (v -> element(&out)){return out;}
            this -> coms -> message(this -> emsx + v -> as_string() + " -> " + _name);
            return out; 
        }

        void print(); 
        void import_model(model_template* _mdl); 
        void import_graphs(std::vector<graph_t*>* grx); 
        void import_mapping(std::map<graph_enum, std::vector<std::string>> mapping); 
        bool next(); 

    private: 
        friend metric_template; 
        friend analysis; 
        template<typename U> friend struct router_t;

        void build(); 
        void getPrediction(); 

        size_t nx = 0; 
        size_t ny = 0; 
        mode_enum _mode; 

        model_template* mdlx = nullptr;
        std::string*     pth = nullptr;
        graph_t*        gr_i = nullptr; 
        tracing_t*      coms = nullptr;

        const std::string emsg = "metric_t::Variable not found: ";
        const std::string emsx = "metric_t::Expected Type: ";

        std::vector<graph_t*>*                   batch_graphs = nullptr; 
        std::vector<std::string*>*                batch_files = nullptr;  
        std::map<graph_enum, std::vector<std::string>>*  vars = nullptr; 
        std::map<graph_enum, std::vector<variable_t*>>* handl = nullptr; 

        std::map<graph_enum, std::map<std::string, size_t>> v_maps = {}; 
        std::map<graph_enum, std::map<std::string, bool>>   h_maps = {}; 
}; 





class metric_model_t : 
    public tools, 
    public notification

{
    public:

        metric_model_t();
        ~metric_model_t(); 

        bool verify(); 

        int kfold = -1;
        int epoch = -1;
        int device = -1;
     
        std::string        run_name = "";  
        std::string checkpoint_path = ""; 
        
        std::map< mode_enum , std::vector<graph_t*>* >     batches = {}; 
        std::map< graph_enum, std::vector<std::string> > variables = {}; 

        model_template*     model = nullptr; 
        torch::TensorOptions* dev = nullptr; 

        metric_t*          metric = nullptr; 
        metric_template*    metrx = nullptr; 
}; 

class metric_template: 
    public tools, 
    public notification

{
    public:
        metric_template(); 
        virtual ~metric_template(); 
        virtual void define_metric(   metric_t* v); 
        virtual void define_variables(metric_t* v); 
        virtual void define_variables(); 
        virtual void start(metric_t*); 
        virtual void end(); 

        virtual metric_template* clone(); 

        std::vector<int> get_kfolds(); 
        cproperty<std::string, metric_template>                             name; 
        cproperty<std::vector<std::string>, metric_template>           variables; 
        cproperty<std::map<std::string, std::string>, metric_template> run_names; 

        std::string output_path = ""; 
        std::string _name = "metric-template"; 

        // --------------------------- functions --------------------------- //
        template <typename T>
        void register_output(std::string tree, std::string __name, T* t){ 
            if (this -> handle){return this -> handle -> process(&tree, &__name, t);}
            this -> handle = new writer();
            this -> handle -> create(this -> output_path); 
            this -> handle -> process(&tree, &__name, t); 
        }

        template <typename T>
        void register_output(router_t<T>* t){ 
            this -> register_output(t -> tr_name + "_training"  , t -> lf_name, t -> get(mode_enum::training  )); 
            this -> register_output(t -> tr_name + "_validation", t -> lf_name, t -> get(mode_enum::validation)); 
            this -> register_output(t -> tr_name + "_evaluation", t -> lf_name, t -> get(mode_enum::evaluation)); 
        }

        template <typename T>
        void write(std::string tree, std::string __name, T* t, bool fill = false){
            if (!this -> handle){return;}
            this -> handle -> process(&tree, &__name, t); 
            if (!fill){return;}
            this -> handle -> write(&tree);
        }

        template <typename g, typename k>
        void sum(std::vector<g*>* ch, k** out){
            k* prt = new k(); 
            prt -> _is_marked = true; 
            std::map<std::string, bool> maps; 
            for (size_t x(0); x < ch -> size(); ++x){
                if (!ch -> at(x)){continue;}
                if (maps[ch -> at(x) -> hash]){continue;}
                maps[ch -> at(x) -> hash] = true;
                prt -> iadd(ch -> at(x));
                prt -> register_child(ch -> at(x)); 
            }
            std::string hash_ = prt -> hash; 
            this -> garbage[hash_].push_back((particle_template*)prt); 
            (*out) = prt;  
        }

        template <typename g>
        void safe_delete(std::vector<g*>* particles){
            for (size_t x(0); x < particles -> size(); ++x){
                if (particles -> at(x) -> _is_marked){continue;}
                this -> pflush(&particles -> at(x)); 
            }
        }

        template <typename g>
        std::vector<g*> make_unique(std::vector<g*>* inpt){
            std::map<std::string, g*> tmp; 
            for (size_t x(0); x < inpt -> size(); ++x){
                std::string hash_ = (*inpt)[x] -> hash; 
                tmp[hash_] = (*inpt)[x]; 
            } 
   
            typename std::vector<g*> out = {}; 
            typename std::map<std::string, g*>::iterator itr; 
            for (itr = tmp.begin(); itr != tmp.end(); ++itr){out.push_back(itr -> second);}
            return out; 
        }

        template <typename g>
        std::vector<g*> vectorize(std::map<std::string, g*>* in){
            typename std::vector<g*> out = {}; 
            typename std::map<std::string, g*>::iterator itr = in -> begin(); 
            for (; itr != in -> end(); ++itr){out.push_back(itr -> second);}
            return out; 
        }

        std::vector<particle_template*> make_particle(
                std::vector<std::vector<double>>* pt,  std::vector<std::vector<double>>* eta, 
                std::vector<std::vector<double>>* phi, std::vector<std::vector<double>>* energy
        ); 

    private:
        friend analysis;
        writer* handle = nullptr; 
        std::vector<metric_model_t*>* data = nullptr; 
        std::map<std::string, writer*> _handles = {}; 
         
        std::string outdir  = "";
        std::map<std::string, std::string>                                  _variables = {}; 
        std::map<std::string, std::string>                                  _run_names = {}; 
        std::map<std::string, model_template*>                                    lnks = {}; 
        std::map<graph_enum, std::vector<std::string>>                       _var_type = {}; 
        std::map<std::string, std::vector<particle_template*>>                 garbage = {}; 
        std::map<std::string, std::map<int, std::map<int, size_t>>>       _epoch_kfold = {};

        void flush_garbage(); 
        bool link(model_template*);
        virtual metric_template* clone(int i); 

        std::map<int, torch::TensorOptions*> get_devices(); 

        void static set_name(std::string*, metric_template*); 
        void static get_name(std::string*, metric_template*);

        void static set_run_name(std::map<std::string, std::string>*, metric_template*); 
        void static get_run_name(std::map<std::string, std::string>*, metric_template*);

        void static set_variables(std::vector<std::string>*, metric_template*); 
        void static get_variables(std::vector<std::string>*, metric_template*);
        void static execute(metric_model_t* mtx, tracing_t* tr); 


}; 

template <typename T>
void router_t<T>::write(metric_template* mtx, metric_t* mti, T* inpt, bool fill){
    T* val = this -> get(mti -> _mode);  
    std::string md = mti -> mode(); 
    *val = *inpt; 
    mtx -> write(this -> tr_name + "_" + md, this -> lf_name, val, fill);
}

#endif
