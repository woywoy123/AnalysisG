#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <structs/switchboards.h>

void metric_template::execute(metric_model_t* mtx, tracing_t* tr){
    auto lambd =[](
        metric_t* mx_, metric_model_t* mtx_, 
        metric_template* mt_, mode_enum md
    ) -> void {
        if (!mtx_ -> batches[md]){return;}
        std::string val = "Starting "; 
        switch(md){
            case mode_enum::training:   val += "Training";   break; 
            case mode_enum::validation: val += "Validation"; break; 
            case mode_enum::evaluation: val += "Evaluation"; break; 
            default: return; 
        }

        mx_ -> _mode = md; 
        mx_ -> coms -> info(val + " Loop");
        mx_ -> import_graphs(mtx_ -> batches[md]); 
        if (!mt_ -> filename.size()){mt_ -> define_variables(mx_);}
        mt_ -> start(mx_); 
        while (mx_ -> next()){
            mt_ -> define_metric(mx_);
            mt_ -> flush_garbage(); 
        }
        mt_ -> end(); 
    }; 

    metric_template*   mt = mtx -> metrx; 
    model_template*   mdl = mtx -> model -> clone(1);
    mdl -> model_checkpoint_path = mtx -> checkpoint_path; 
    mdl -> restore_state(); 

    tools::replace(&mtx -> run_name, "::", "/"); 

    metric_t* mx = mtx -> metric; 
    mx -> kfold  = mtx -> kfold; 
    mx -> epoch  = mtx -> epoch;
    mx -> run_name = mtx -> run_name; 
    mx -> import_mapping(mtx -> variables); 
    mx -> import_model(mdl); 

    mx -> coms = tr; 
    lambd(mx, mtx, mt, mode_enum::training); 
    lambd(mx, mtx, mt, mode_enum::validation); 
    lambd(mx, mtx, mt, mode_enum::evaluation); 
    if (mt -> handle){mt -> handle -> close();}
    //tr -> success("FINISHED!"); 
    tools::pflush(&mdl); 
    tools::pflush(&mtx -> metric);
    tr -> finished(); 

}

