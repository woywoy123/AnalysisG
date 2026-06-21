#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <meta/meta.h>

metric_template::metric_template(){
    this -> name.set_object(this); 
    this -> name.set_setter(this -> set_name); 
    this -> name.set_getter(this -> get_name); 

    this -> run_names.set_object(this); 
    this -> run_names.set_setter(this -> set_run_name); 
    this -> run_names.set_getter(this -> get_run_name); 

    this -> variables.set_object(this); 
    this -> variables.set_setter(this -> set_variables); 
    this -> variables.set_getter(this -> get_variables); 

    this -> output_path.set_object(this);
    this -> output_path.set_getter(this -> get_output); 
}

metric_template::~metric_template(){
    this -> handle = nullptr; 
    std::map<std::string, writer*>::iterator wrt = this -> _handles.begin();
    for (; wrt != this -> _handles.end(); ++wrt){
        if (!wrt -> second){continue;}
        delete wrt -> second; 
        this -> _handles[wrt -> first] = nullptr;
    }
    this -> _handles.clear(); 
}

metric_template* metric_template::clone(){return new metric_template();}
metric_template* metric_template::clone(int){
    metric_template* mx = this -> clone(); 
    mx -> _var_type     = this -> _var_type;
    mx -> _epoch_kfold  = this -> _epoch_kfold;
    return mx; 
}

void metric_template::define_metric(metric_t*){}
void metric_template::define_variables(){}

void metric_template::dynamic_write(std::string v){
    if (!v.size()){v = this -> name;}
    this -> _outdir = this -> base_pth + v + ".root";
    if (!this -> _handles.count(v)){this -> handle = nullptr;}
    if (!this -> handle){
        this -> define_variables();
        this -> _handles[v] = this -> handle;
    }
    else {this -> handle = this -> _handles[v];} 
}

void metric_template::event(){}; 
void metric_template::batch(){}; 
void metric_template::end(){}; 
void metric_template::start(metric_t*){}; 
