#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <structs/switchboards.h>

void metric_template::execute(metric_model_t* mtx, tracing_t* tr){
    metric_template*   mt = mtx -> metrx; 
    model_template*   mdl = mtx -> model -> clone(1);
    mdl -> model_checkpoint_path = mtx -> checkpoint_path; 
    mdl -> restore_state(); 

    metric_t* mx = mtx -> metric; 
    mx -> kfold  = mtx -> kfold; 
    mx -> epoch  = mtx -> epoch;
    mx -> import_mapping(mtx -> variables); 
    mx -> import_model(mdl); 

    std::map< mode_enum , std::vector<graph_t*>* > batches = mtx -> batches;

    mt -> define_variables(mx); 
    mx -> _mode = mode_enum::training; 
    mx -> import_graphs(batches[mode_enum::training]); 
    mx -> coms = tr; 
    mt -> start(mx); 
    while (mx -> next()){
        mt -> define_metric(mx);
        mt -> flush_garbage(); 
    }

    mx -> _mode = mode_enum::validation;
    mx -> import_graphs(batches[mode_enum::validation]); 
    while (mx -> next()){
        mt -> define_metric(mx); 
        mt -> flush_garbage(); 
    }

    mx -> _mode = mode_enum::evaluation; 
    mx -> import_graphs(batches[mode_enum::evaluation]); 
    while (mx -> next()){
        mt -> define_metric(mx); 
        mt -> flush_garbage(); 
    }
    mt -> end(); 
    if (mt -> handle){mt -> handle -> close();}
    tools::pflush(&mdl); 
    tools::pflush(&mtx->metric); 
    tr -> finished(); 

}

