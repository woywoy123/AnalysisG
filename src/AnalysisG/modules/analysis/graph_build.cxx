#include <generators/analysis.h>

void analysis::build_graphs(){
    this -> success("+============================+"); 
    this -> success("|   Starting Graph Builder   |");
    this -> success("+============================+"); 

    std::map<std::string, std::map<std::string, graph_template*>>::iterator itx; 
    for (itx = this -> graph_labels.begin(); itx != this -> graph_labels.end(); ++itx){
        std::string label = itx -> first; 
        std::vector<event_template*>* events_ = this -> tracer -> get_events(label); 
        std::map<std::string, graph_template*>::iterator itx_; 

        for (itx_ = itx -> second.begin(); itx_ != itx -> second.end(); ++itx_){
            for (event_template* ev : *events_){
                std::vector<std::string> spl = this -> split(ev -> filename, "/"); 
                std::string fname = this -> hash(ev -> filename) + "-" + spl[spl.size()-1]; 
                this -> replace(&fname, ".root", ".h5"); 
                fname = itx_ -> first + "/" + fname; 
                if (this -> in_cache[ev -> filename][fname]){continue;}

                graph_template* gr_o = itx_ -> second -> build(ev); 
                bool rm = this -> tracer -> add_graph(gr_o, label);
                if (!rm){continue;}
                delete gr_o;
            }
        }
        delete events_; 
    }
    this -> success("Finished Building Graphs from events"); 
}

void analysis::build_dataloader(bool training){
    if (!this -> loader -> data_set -> size()){
        this -> tracer -> populate_dataloader(this -> loader); 
    }
    if (!training){return;}

    std::string path_data = this -> m_settings.training_dataset; 
    if (path_data.size() && !this -> loader -> restore_dataset(path_data)){
        this -> loader -> generate_test_set(this -> m_settings.train_size);
        this -> loader -> generate_kfold_set(this -> m_settings.kfolds); 
        this -> loader -> dump_dataset(path_data);
    }
    if (!path_data.size()){
        this -> loader -> generate_test_set(this -> m_settings.train_size);
        this -> loader -> generate_kfold_set(this -> m_settings.kfolds); 
    }
}
