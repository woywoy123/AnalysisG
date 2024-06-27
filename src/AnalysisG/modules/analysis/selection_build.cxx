#include <generators/analysis.h>

void analysis::build_selections(){
    std::map<std::string, selection_template*>::iterator itx = this -> selection_names.begin(); 
    for (; itx != this -> selection_names.end(); ++itx){
        std::vector<event_template*>* events_ = nullptr; 
        events_ = this -> tracer -> get_events(""); 
        if (!events_ -> size()){
            this -> warning("No Events found for Selection. Skipping...");
            continue;
        }

        selection_template* sel_t = itx -> second; 
        for (size_t x(0); x < events_ -> size(); ++x){
            selection_template* sel_o = sel_t -> build(events_ -> at(x)); 
            bool rm = this -> tracer -> add_selection(sel_o);
            if (!rm){continue;}
            delete sel_o;
        }
        delete events_; 
    }
    this -> success("Built Selections from Events"); 
}




