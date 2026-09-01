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
    this -> vflush( this -> data); 
    this -> pflush(&this -> data);  
}

metric_template* metric_template::clone(){return new metric_template();}

metric_template* metric_template::clone(int i){
    metric_template* mtx = this -> clone(); 
    mtx -> lnks          = this -> lnks; 
    mtx -> output_path   = this -> output_path; 
    mtx -> _var_type     = this -> _var_type; 
    mtx -> _variables    = this -> _variables; 
    mtx -> _run_names    = this -> _run_names; 
    mtx -> _epoch_kfold  = this -> _epoch_kfold;

    if (i == 1){
        this -> lnks.clear(); 
        std::vector<metric_model_t*>* v = this -> data; 
        this -> data = mtx -> data; mtx -> data = v; 
    }
    return mtx;
}

void metric_template::define_metric(metric_t*){}
void metric_template::define_variables(metric_t*){}; 
void metric_template::define_variables(){}; 
void metric_template::end(){}; 
void metric_template::start(metric_t*){}; 

