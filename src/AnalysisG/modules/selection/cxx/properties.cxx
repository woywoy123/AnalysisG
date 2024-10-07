#include <templates/selection_template.h>

void selection_template::set_name(std::string* name, selection_template* ev){
    ev -> data.name = *name; 
}

void selection_template::get_name(std::string* name, selection_template* ev){
    *name = ev -> data.name; 
}

void selection_template::set_hash(std::string* path, selection_template* ev){
    if (ev -> data.hash.size()){return;}
    std::string x = *path + "/" + ev -> to_string(ev -> index); 
    ev -> data.hash = ev -> tools::hash(x, 18); 
    *path = ""; 
}

void selection_template::get_hash(std::string* val, selection_template* ev){
    *val = ev -> data.hash; 
}

void selection_template::get_tree(std::string* name, selection_template* ev){
    *name = ev -> data.tree; 
}

void selection_template::set_weight(double* inpt, selection_template* ev){
    ev -> data.weight = *inpt; 
}

void selection_template::get_weight(double* inpt, selection_template* ev){
    *inpt = ev -> data.weight; 
}

void selection_template::set_index(long* inpt, selection_template* ev){
    ev -> data.index = *inpt; 
}


