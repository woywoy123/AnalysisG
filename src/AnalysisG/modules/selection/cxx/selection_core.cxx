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
    std::map<std::string, std::vector<particle_template*>>::iterator itr;
    for (itr = this -> garbage.begin(); itr != this -> garbage.end(); ++itr){
        for (size_t x(0); x < itr -> second.size(); ++x){
            if (!itr -> second[x] -> _is_marked){continue;}
            delete itr -> second[x];
        }
        itr -> second.clear(); 
    }
    this -> garbage.clear();
}

bool selection_template::operator == (selection_template& p){return this -> hash == p.hash;}


bool selection_template::CompileEvent(){
    if (!this -> selection(this -> m_event)){return false;}
    if (!this -> strategy(this -> m_event)){return false;}
    return true; 
}

selection_template* selection_template::build(event_template* ev){
    selection_template* sel = this -> clone(); 
    std::string _name = sel -> name; 
    sel -> m_event  = ev; 
    sel -> data     = ev -> data; 
    sel -> name     = _name; 
    sel -> weight   = ev -> weight; 
    sel -> filename = ev -> filename;
    return sel; 
}

void selection_template::bulk_write_out(){
    if (!this -> p_bulk_write || !this -> handle){return;}
    std::unordered_map<long, std::string>::iterator itx = this -> sequence.begin(); 
    for (; itx != this -> sequence.end(); ++itx){
        this -> bulk_write(&itx -> first, &itx -> second); 
        this -> handle -> write(); 
    }
}

void selection_template::merger(selection_template* sl2){
    if (this -> name != sl2 -> name){return;}
    this -> merge(sl2); 
    if (this -> m_event){return;}

    if (sl2 -> m_event){
        if (this -> p_bulk_write){this -> sequence[sl2 -> index] = sl2 -> hash;}
        else {this -> write(float(sl2 -> weight), "event_weight");}
        this -> passed_weights[sl2 -> filename][sl2 -> hash] = sl2 -> weight;
        this -> matched_meta[sl2 -> filename] = sl2 -> meta_data -> meta_data;
        return; 
    }

    merge_data(&this -> passed_weights, &sl2 -> passed_weights);
    merge_data(&this -> matched_meta  , &sl2 -> matched_meta); 
} 

std::vector<std::map<std::string, float>> selection_template::reverse_hash(std::vector<std::string>* hashes){
    std::vector<std::map<std::string, float>> output; 
    output.assign(hashes -> size(), {{"None", 0}}); 
    for (size_t x(0); x < hashes -> size(); ++x){
        std::string _hash = (*hashes)[x]; 
        std::map<std::string, std::map<std::string, float>>::iterator itr;
        for (itr = this -> passed_weights.begin(); itr != this -> passed_weights.end(); ++itr){
            if (!itr -> second.count(_hash)){continue;}
            output[x] = {{itr -> first, itr -> second[_hash]}}; 
            break; 
        }
    }
    return output; 
}

void selection_template::switch_board(particle_enum attrs, particle_template* ptr, std::vector<int>* _data){
    switch (attrs){
        case particle_enum::pdgid: return _data -> push_back(ptr -> pdgid);
        case particle_enum::index: return _data -> push_back(ptr -> index);
        default: return; 
    }
}

void selection_template::switch_board(particle_enum attrs, particle_template* ptr, std::vector<double>* _data){
    switch (attrs){
        case particle_enum::pt:     return _data -> push_back(ptr -> pt); 
        case particle_enum::eta:    return _data -> push_back(ptr -> eta); 
        case particle_enum::phi:    return _data -> push_back(ptr -> phi); 
        case particle_enum::energy: return _data -> push_back(ptr -> e); 
        case particle_enum::px:     return _data -> push_back(ptr -> px); 
        case particle_enum::py:     return _data -> push_back(ptr -> py); 
        case particle_enum::pz:     return _data -> push_back(ptr -> pz); 
        case particle_enum::mass:   return _data -> push_back(ptr -> mass); 
        case particle_enum::charge: return _data -> push_back(ptr -> charge); 
        default: return; 
    }
}

void selection_template::switch_board(particle_enum attrs, particle_template* ptr, std::vector<bool>* _data){
    switch (attrs){
        case particle_enum::is_b:   return _data -> push_back(ptr -> is_b); 
        case particle_enum::is_lep: return _data -> push_back(ptr -> is_lep); 
        case particle_enum::is_nu:  return _data -> push_back(ptr -> is_nu); 
        case particle_enum::is_add: return _data -> push_back(ptr -> is_add); 
        default: return; 
    }
}

void selection_template::switch_board(particle_enum attrs, particle_template* ptr, std::vector<std::vector<double>>* _data){
    std::vector<double> tmp; 
    switch (attrs){
        case particle_enum::pmc: 
                this -> switch_board(particle_enum::px    , ptr, &tmp); 
                this -> switch_board(particle_enum::py    , ptr, &tmp);
                this -> switch_board(particle_enum::pz    , ptr, &tmp);
                this -> switch_board(particle_enum::energy, ptr, &tmp); 
                break; 
        case particle_enum::pmu:
                this -> switch_board(particle_enum::pt    , ptr, &tmp); 
                this -> switch_board(particle_enum::eta   , ptr, &tmp); 
                this -> switch_board(particle_enum::phi   , ptr, &tmp); 
                this -> switch_board(particle_enum::energy, ptr, &tmp); 
                break; 
        default: return; 
    }
    _data -> push_back(tmp); 
}

