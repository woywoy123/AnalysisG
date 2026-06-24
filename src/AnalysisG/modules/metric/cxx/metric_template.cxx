#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <meta/meta.h>

metric_template::metric_template(){
    this -> data = new std::vector<metric_model_t*>(); 

    this -> name.set_object(this); 
    this -> name.set_setter(this -> set_name); 
    this -> name.set_getter(this -> get_name); 

    this -> run_names.set_object(this); 
    this -> run_names.set_setter(this -> set_run_name); 
    this -> run_names.set_getter(this -> get_run_name); 

    this -> variables.set_object(this); 
    this -> variables.set_setter(this -> set_variables); 
    this -> variables.set_getter(this -> get_variables); 
}

metric_template::~metric_template(){
    this -> handle -> close(); 
    delete this -> handle; 
    this -> vflush( this -> data); 
    this -> pflush(&this -> data);  
    this -> mflush(&this -> _handles); 
}

metric_template* metric_template::clone(){return new metric_template();}

metric_template* metric_template::clone(int i){
    metric_template* mtx = this -> clone(); 
    mtx -> outdir        = this -> outdir;
    mtx -> lnks          = this -> lnks; this -> lnks.clear(); 

    mtx -> _var_type     = this -> _var_type; 
    mtx -> _variables    = this -> _variables; 
    mtx -> _run_names    = this -> _run_names; 
    mtx -> _epoch_kfold  = this -> _epoch_kfold;

    std::vector<metric_model_t*>* v = this -> data; 
    this -> data = mtx -> data; 
    mtx -> data = v; 
    return mtx;
}

void metric_template::define_metric(metric_t*){}
void metric_template::define_variables(metric_t*){}; 
void metric_template::define_variables(){}; 

void metric_template::event(){}; 
void metric_template::batch(){}; 
void metric_template::end(){}; 
void metric_template::start(metric_t*){}; 

void metric_template::get_variables(std::vector<std::string>* rn_name, metric_template* ev){ 
    std::map<std::string, std::string>::iterator itx = ev -> _variables.begin(); 
    for (; itx != ev -> _variables.end(); ++itx){rn_name -> push_back(itx -> first);}
}

void metric_template::get_run_name(
        std::map<std::string, std::string>* rn_name, metric_template* ev
){ 
    *rn_name = ev -> _run_names; 
}

std::vector<int> metric_template::get_kfolds(){
    std::vector<int> out = {}; 
    std::map<int, bool> oux = {}; 
    for (size_t x(0); x < this -> data -> size(); ++x){
        metric_model_t* wrk = this -> data -> at(x); 
        if (oux[wrk -> kfold]){continue;}
        if (wrk -> kfold < 0){continue;}
        oux[wrk -> kfold] = true;
        out.push_back(wrk -> kfold); 
    }
    return out; 
}
