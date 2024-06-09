#include <templates/event_template.h>

event_template::event_template(){
    this -> trees.set_setter(this -> set_trees); 
    this -> trees.set_object(this); 

    this -> branches.set_setter(this -> set_branches); 
    this -> branches.set_object(this); 

    this -> leaves.set_getter(this -> get_leaves); 
    this -> leaves.set_object(this); 

    this -> name.set_setter(this -> set_name); 
    this -> name.set_object(this); 

    this -> hash.set_setter(this -> set_hash); 
    this -> hash.set_getter(this -> get_hash); 
    this -> hash.set_object(this); 

    this -> tree.set_setter(this -> set_tree); 
    this -> tree.set_getter(this -> get_tree); 
    this -> tree.set_object(this); 

    this -> weight.set_setter(this -> set_weight); 
    this -> weight.set_object(this); 

    this -> index.set_setter(this -> set_index); 
    this -> index.set_object(this); 
}

event_template::~event_template(){
    std::map<std::string, particle_template*>::iterator itrp = this -> particle_generators.begin();
    for (; itrp != this -> particle_generators.end(); ++itrp){delete itrp -> second;}
}

bool event_template::operator == (event_template& p){
    return this -> hash == p.hash; 
}

void event_template::build_mapping(std::map<std::string, data_t*> evnt){
    if (this -> tree_variable_link.size()){return;}

    std::vector<std::string> tr = this -> trees; 
    for (int x(0); x < tr.size(); ++x){
        std::map<std::string, data_t*>::iterator ite = evnt.begin(); 
        for (; ite != evnt.end(); ++ite){
            bool s = tr[x] == ite -> second -> tree_name;
            if (!s){continue;}

            std::map<std::string, std::string>::iterator itl = this -> m_leaves.begin(); 
            for (; itl != this -> m_leaves.end(); ++itl){
                bool var = ite -> second -> leaf_name == itl -> second; 
                if (!var){continue;}

                std::vector<std::string> type_name = this -> split(itl -> first, "/"); 
                this -> tree_variable_link[tr[x]][type_name[0]].tree = tr[x]; 
                this -> tree_variable_link[tr[x]][type_name[0]].handle[type_name[1]] = ite -> second; 
                break; 
            } 
        }
    }
}

std::map<std::string, event_template*> event_template::build_event(std::map<std::string, data_t*> evnt){
    this -> build_mapping(evnt); 
    bool next_ = true; 
    std::map<std::string, event_template*> output = {}; 
    std::map<std::string, std::map<std::string, element_t>>::iterator itr = this -> tree_variable_link.begin();
    for (; itr != this -> tree_variable_link.end(); ++itr){
        event_template* ev = this -> clone(); 
        ev -> tree = itr -> first; 

        std::map<std::string, element_t>::iterator itrx = this -> tree_variable_link[itr -> first].begin(); 
        for (; itrx != this -> tree_variable_link[itr -> first].end(); ++itrx){
            if (itrx -> first == std::string(this -> name)){ev -> build(&itrx -> second);}
            else {ev -> particle_generators[itrx -> first] -> build(ev -> particle_link[itrx -> first], &itrx -> second);}
            next_ *= this -> tree_variable_link[itr -> first][itrx -> first].next(); 
        } 
        if (next_){output[itr -> first] = ev;}
        else {delete ev;}
    }
    return output; 
}

void event_template::build(element_t* el){
    return; 
}

event_template* event_template::clone(){
    return new event_template(); 
}

