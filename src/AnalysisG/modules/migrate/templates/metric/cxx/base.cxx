#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <structs/switchboards.h>

void metric_template::execute(metric_model_t* mtx, size_t* prg, std::string* msg){
    model_template*   mdl = mtx -> model -> clone(1);
    mdl -> model_checkpoint_path = mtx -> checkpoint_path; 
    mdl -> restore_state(); 
   
    metric_t* mx = mtx -> metric; 
    mx -> kfold  = mtx -> kfold; 
    mx -> epoch  = mtx -> epoch;

    mx -> import_model(mdl); 
    mx -> import_mapping(mtx -> variables); 

    std::map< mode_enum , std::vector<graph_t*>* > batches = mtx -> batches;

    metric_template*  mt = mtx -> metrx -> clone(); 

    mt -> define_variables(mx); 
    mx -> _mode = mode_enum::training; 
    mx -> import_graphs(batches[mode_enum::training]); 
    while (mx -> next()){
        mt -> start(mx); 
        mt -> define_metric(mx); (*prg)++; 
        mt -> flush_garbage(); 
    }

    mx -> _mode = mode_enum::validation;
    mx -> import_graphs(batches[mode_enum::validation]); 
    while (mx -> next()){
        mt -> start(mx); 
        mt -> define_metric(mx); (*prg)++;  
        mt -> flush_garbage(); 
    }

    mx -> _mode = mode_enum::evaluation; 
    mx -> import_graphs(batches[mode_enum::evaluation]); 
    while (mx -> next()){
        mt -> start(mx); 
        mt -> define_metric(mx); (*prg)++;  
        mt -> flush_garbage(); 
    }
    mt -> end(); 
    (*prg) = 1; 
    tools::pflush(&mdl); 
    tools::pflush(&mt); 
    //tools::pflush(&mx); 

}

