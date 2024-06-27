#include <templates/selection_template.h>

selection_template::selection_template(){
    this -> name.set_object(this); 
    this -> name.set_setter(this -> set_name); 
    this -> name.set_getter(this -> get_name); 

    this -> hash.set_setter(this -> set_hash); 
    this -> hash.set_getter(this -> get_hash); 
    this -> hash.set_object(this); 

    this -> tree.set_object(this); 
    this -> tree.set_getter(this -> get_tree); 

    this -> weight.set_setter(this -> set_weight); 
    this -> weight.set_object(this); 

    this -> index.set_setter(this -> set_index); 
    this -> index.set_object(this); 
    this -> name = "selection-template"; 
}

selection_template::~selection_template(){}

bool selection_template::operator == (selection_template& p){
    return this -> hash == p.hash; 
}

bool selection_template::selection(event_template* ev){
    return true; 
}

bool selection_template::strategy(event_template* ev){
    return true; 
}

void selection_template::CompileEvent(){
    if (!this -> selection(this -> m_event)){return;}
    if (!this -> strategy(this -> m_event)){return;}
    return; 
}

selection_template* selection_template::build(event_template* ev){
    event_t* data_ = &ev -> data;
    data_ -> name = this -> name; 
    selection_template* sel = this -> clone(); 
    sel -> m_event = ev; 
    sel -> data = *data_; 
    sel -> filename = ev -> filename; 
    return sel; 
}

selection_template* selection_template::clone(){
    return new selection_template(); 
}

void selection_template::merge(selection_template* sl2){}

void selection_template::merger(selection_template* sl2){
    if (this -> name != sl2 -> name){return;}
    this -> merge(sl2); 
} 

