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
        std::string* get_filename(long unsigned int idx); 

    private: 
        friend metric_template; 
        friend analysis; 

        void build(); 
        mode_enum train_mode; 
        std::string* pth = nullptr;
        model_template* mdlx = nullptr;
        metric_template* mtx = nullptr;
        size_t index = 0; 
        
        std::vector<std::string*>* batch_files = nullptr;  
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
        virtual void define_variables(); 
        virtual void define_metric(metric_t* v); 
        virtual void event();
        virtual void batch();
        virtual void end(); 

        cproperty<std::string, metric_template> name; 
        cproperty<std::string, metric_template> output_path; 
        cproperty<std::vector<std::string>, metric_template> variables; 
        cproperty<std::map<std::string, std::string>, metric_template> run_names; 

        // --------------------------- functions --------------------------- //
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
            }
            std::string hash_ = prt -> hash; 
            this -> garbage[hash_].push_back((particle_template*)prt); 
            (*out) = prt;  
        }

        template <typename g>
        void safe_delete(std::vector<g*>* particles){
            for (size_t x(0); x < particles -> size(); ++x){
                if (particles -> at(x) -> _is_marked){continue;}
                delete particles -> at(x); 
                (*particles)[x] = nullptr; 
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

        std::map<std::string, model_template*> lnks; 
        std::map<std::string, std::vector<model_template*>> hash_mdl = {}; 
        std::map<std::string, std::map<mode_enum, std::vector<graph_t*>*>> hash_bta = {}; 
        std::map<std::string, std::map<int, std::map<int, std::string>>> _epoch_kfold;
        std::map<std::string, std::map<graph_enum, std::vector<std::string>>> _var_type; 
        
        std::string _name = "metric-template"; 
        std::string _outdir = ""; 

        std::map<std::string, std::string> _run_names = {}; 
        std::map<std::string, std::string> _variables = {}; 

        void flush_garbage(); 

        void static set_name(std::string*, metric_template*); 
        void static get_name(std::string*, metric_template*);

        void static set_run_name(std::map<std::string, std::string>*, metric_template*); 
        void static get_run_name(std::map<std::string, std::string>*, metric_template*);

        void static set_variables(std::vector<std::string>*, metric_template*); 
        void static get_variables(std::vector<std::string>*, metric_template*);

        void static get_output(std::string* out, metric_template* ev); 

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

        std::map<std::string, std::vector<particle_template*>> garbage = {}; 
        writer* handle = nullptr; 
}; 


#endif
