#include <generators/analysis.h>

void analysis::build_graphs(){
    std::map<std::string, graph_template*>::iterator itx_; 
    std::map<std::string, std::map<std::string, graph_template*>>::iterator itx; 

    itx = this -> graph_labels.begin(); 
    for (; itx != this -> graph_labels.end(); ++itx){
        std::string label = itx -> first; 
        std::vector<event_template*>* events_ = nullptr; 
        events_ = this -> tracer -> get_events(label); 
        if (!events_ -> size()){
            this -> warning("No Events found for Graph (" + label + "). Skipping...");
            continue;
        }

        itx_ = itx -> second.begin(); 
        for (; itx_ != itx -> second.end(); ++itx_){
            graph_template* gr_t = itx_ -> second; 
            long ls = events_ -> size(); 
            for (size_t x(0); x < ls; ++x){
                graph_template* gr_o = gr_t -> build(events_ -> at(x)); 
                bool rm = this -> tracer -> add_graph(gr_o, label);
                if (!rm){continue;}
                delete gr_o;
            }
        }
        delete events_; 
    }
    this -> success("Built Graphs from Events"); 
}

void analysis::build_dataloader(){
    this -> tracer -> populate_dataloader(this -> loader); 
    this -> loader -> generate_test_set(this -> train_size);
    this -> loader -> generate_kfold_set(this -> kfolds); 
}






