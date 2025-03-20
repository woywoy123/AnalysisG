#include <generators/analysis.h>

void analysis::build_selections(){
    std::vector<event_template*> events_ = this -> tracer -> get_events(""); 
    if (!events_.size()){return this -> warning("No Events found for Selection. Skipping...");}
    if (this -> m_settings.selection_root){this -> tracer -> output_path = &this -> m_settings.output_path;}
    std::map<std::string, selection_template*>::iterator itx = this -> selection_names.begin(); 
    for (; itx != this -> selection_names.end(); ++itx){
        selection_template* sel_t = itx -> second; 
        for (size_t x(0); x < events_.size(); ++x){
            selection_template* sel_o = sel_t -> build(events_[x]); 
            if (!this -> tracer -> add_selection(sel_o)){continue;}
            delete sel_o;
        }
    }
    this -> success("Built Selections from Events"); 
}




