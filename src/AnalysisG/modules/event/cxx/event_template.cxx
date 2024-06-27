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
    if (this -> filename.size()){this -> flush_particles();}
    std::map<std::string, particle_template*>::iterator itrp = this -> particle_generators.begin();
    for (; itrp != this -> particle_generators.end(); ++itrp){delete itrp -> second;} 
}

bool event_template::operator == (event_template& p){
    return this -> hash == p.hash; 
}

void event_template::build_mapping(std::map<std::string, data_t*>* evnt){
    if (this -> tree_variable_link.size()){return;}

    std::vector<std::string> tr = this -> trees; 
    for (int x(0); x < tr.size(); ++x){
        this -> next_[tr[x]] = true; 
        std::map<std::string, data_t*>::iterator ite = evnt -> begin(); 
        for (; ite != evnt -> end(); ++ite){
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
    this -> particle_link.clear(); 
}

std::map<std::string, event_template*> event_template::build_event(std::map<std::string, data_t*>* evnt){
    this -> build_mapping(evnt); 
    std::map<std::string, event_template*> output = {}; 
    std::map<std::string, std::map<std::string, element_t>>::iterator itr = this -> tree_variable_link.begin();
    for (; itr != this -> tree_variable_link.end(); ++itr){
        if (!this -> next_[itr -> first]){continue;}
        event_template* ev = this -> clone(); 
        ev -> tree = itr -> first; 

        std::map<std::string, element_t>::iterator itrx = this -> tree_variable_link[itr -> first].begin(); 
        for (; itrx != this -> tree_variable_link[itr -> first].end(); ++itrx){
            itrx -> second.set_meta(); 

            if (itrx -> first == std::string(ev -> name)){ev -> build(&itrx -> second);}
            else {
                std::map<std::string, particle_template*>* builder = new std::map<std::string, particle_template*>(); 
                ev -> particle_generators[itrx -> first] -> build(builder, &itrx -> second);
                std::map<std::string, particle_template*>* m_link = ev -> particle_link[itrx -> first]; 
                m_link -> insert(builder -> begin(), builder -> end()); 
                ev -> particle_link[itrx -> first] = builder; 
            }
            this -> next_[itr -> first] *= this -> tree_variable_link[itr -> first][itrx -> first].next(); 
            ev -> index = itrx -> second.event_index; 
            ev -> hash = itrx -> second.filename; 
            ev -> filename = itrx -> second.filename;
        } 
        ev -> flush_leaf_string(); 
        output[itr -> first] = ev;
    }
    return output; 
}

void event_template::build(element_t* el){
    return; 
}

void event_template::flush_leaf_string(){
    this -> m_trees = {};
    this -> m_branches = {};
    this -> m_leaves = {}; 

    std::map<std::string, std::map<std::string, particle_template*>*>::iterator itr; 
    for (itr = this -> particle_link.begin(); itr != this -> particle_link.end(); ++itr){
        std::map<std::string, particle_template*>* pmap = itr -> second; 
        std::map<std::string, particle_template*>::iterator itx = pmap -> begin();
        for (; itx != pmap -> end(); ++itx){itx -> second -> leaves = {};}
    }
    std::map<std::string, particle_template*>::iterator itrp = this -> particle_generators.begin();
    for (; itrp != this -> particle_generators.end(); ++itrp){
        delete this -> particle_generators[itrp -> first]; 
        this -> particle_generators[itrp -> first] = nullptr; 
    } 
    this -> particle_generators.clear(); 
}

void event_template::flush_particles(){
    std::map<std::string, std::map<std::string, particle_template*>*>::iterator itr; 
    for (itr = this -> particle_link.begin(); itr != this -> particle_link.end(); ++itr){
        std::map<std::string, particle_template*>* pmap = itr -> second; 
        std::map<std::string, particle_template*>::iterator itx = pmap -> begin();
        for (; itx != pmap -> end(); ++itx){
            delete (*pmap)[itx -> first];
            (*pmap)[itx -> first] = nullptr; 
        }
        pmap -> clear(); 
        delete pmap; 
    }
    this -> particle_link = {}; 
}

void event_template::CompileEvent(){
    return; 
}

event_template* event_template::clone(){
    return new event_template(); 
}

