#include <templates/event_template.h>

void event_template::set_trees(std::vector<std::string>* inpt, event_template* ev){
    for (int x(0); x < inpt -> size(); ++x){
        ev -> m_trees[inpt -> at(x)] = "tree"; 
    }
}

void event_template::set_branches(std::vector<std::string>* inpt, event_template* ev){
    for (int x(0); x < inpt -> size(); ++x){
        ev -> m_branches[inpt -> at(x)] = "branch"; 
    }
}

void event_template::add_leaf(std::string key, std::string val){
    std::string n = this -> name; 
    this -> m_leaves[n + "/" + key] = val; 
}

void event_template::get_leaves(std::vector<std::string>* inpt, event_template* ev){
    std::map<std::string, std::string>::iterator itr = ev -> m_leaves.begin(); 
    for (; itr != ev -> m_leaves.end(); ++itr){inpt -> push_back(itr -> second);}
}

void event_template::set_tree(std::string* name, event_template* ev){
    ev -> data.name = *name; 
}

void event_template::get_tree(std::string* name, event_template* ev){
    *name = ev -> data.name; 
}

void event_template::set_name(std::string* name, event_template* ev){
    ev -> data.name = *name; 
}

void event_template::set_hash(std::string* path, event_template* ev){
    std::string x = *path + "/"; 
    x += ev -> to_string(ev -> index); 
    x += std::string(ev -> name); 
    ev -> data.hash = ev -> tools::hash(x, 18); 
}

void event_template::get_hash(std::string* val, event_template* ev){
    *val = ev -> data.hash; 
}

void event_template::set_weight(double* inpt, event_template* ev){
     ev -> data.weight = *inpt; 
}

void event_template::set_index(long* inpt, event_template* ev){
    ev -> data.index = *inpt; 
}


