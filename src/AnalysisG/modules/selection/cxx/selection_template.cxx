#include <templates/selection_template.h>
#include <meta/meta.h>

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
    this -> weight.set_getter(this -> get_weight); 
    this -> weight.set_object(this); 

    this -> index.set_setter(this -> set_index); 
    this -> index.set_object(this); 

    this -> name = "selection-template"; 
}

selection_template::~selection_template(){
    std::map<std::string, std::vector<particle_template*>>::iterator itr = this -> garbage.begin();
    for (; itr != this -> garbage.end(); ++itr){
        for (size_t x(0); x < itr -> second.size(); ++x){delete itr -> second[x];}
    }
    this -> garbage.clear();
}

bool selection_template::operator == (selection_template& p){
    return this -> hash == p.hash; 
}

bool selection_template::selection(event_template* ev){
    return true; 
}

bool selection_template::strategy(event_template* ev){
    return true; 
}

bool selection_template::CompileEvent(){
    if (!this -> selection(this -> m_event)){return false;}
    if (!this -> strategy(this -> m_event)){return false;}
    return true; 
}

selection_template* selection_template::build(event_template* ev){
    selection_template* sel = this -> clone(); 
    std::string name = sel -> name; 
    sel -> m_event  = ev; 
    sel -> data     = ev -> data; 
    sel -> name     = name; 
    sel -> weight   = ev -> weight; 
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
    if (this -> m_event){return;}

    if (sl2 -> m_event){
        this -> passed_weights[sl2 -> filename][sl2 -> hash] = sl2 -> weight;
        this -> matched_meta[sl2 -> filename] = sl2 -> meta_data -> meta_data;
        return; 
    }

    merge_data(&this -> passed_weights, &sl2 -> passed_weights);
    merge_data(&this -> matched_meta, &sl2 -> matched_meta); 
} 

std::vector<std::map<std::string, float>> selection_template::reverse_hash(std::vector<std::string>* hashes){
    std::vector<std::map<std::string, float>> output; 
    output.assign(hashes -> size(), {{"None", 0}}); 
    for (size_t x(0); x < hashes -> size(); ++x){
        std::string hash = (*hashes)[x]; 
        std::map<std::string, std::map<std::string, float>>::iterator itr;
        for (itr = this -> passed_weights.begin(); itr != this -> passed_weights.end(); ++itr){
            if (!itr -> second.count(hash)){continue;}
            output[x] = {{itr -> first, itr -> second[hash]}}; 
            break; 
        }
    }
    return output; 
}





