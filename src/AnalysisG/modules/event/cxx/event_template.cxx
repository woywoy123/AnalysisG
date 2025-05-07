/**
 * @file event_template.cxx
 * @brief Implementation of event_template class methods for physics event data handling.
 * 
 * This file implements the methods declared in the event_template.h header file.
 * It provides functionality for managing event data, including registration and
 * manipulation of particles, trees, branches, and leaves within the event structure.
 */

#include <templates/event_template.h>

/**
 * @brief Constructor for the event_template class.
 * 
 * Initializes a new event_template instance and sets up property accessors
 * for all the class properties. These accessors enable controlled access to
 * the internal state of the event and ensure proper synchronization between
 * the properties and their underlying data.
 */
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

/**
 * @brief Destructor for the event_template class.
 * 
 * Cleans up resources used by the event_template instance.
 * This includes deregistering particles and cleaning up any dynamically
 * allocated memory to prevent memory leaks.
 */
event_template::~event_template(){
    if (this -> filename.size()){this -> flush_particles();}
    this -> deregister_particle(&this -> particle_generators); 
}

/**
 * @brief Equality operator for event_template objects.
 * @param p Reference to another event_template object
 * @return True if the two objects are equal, false otherwise
 * 
 * Compares two event_template objects based on their hash values.
 */
bool event_template::operator == (event_template& p){
    return this -> hash == p.hash; 
}

/**
 * @brief Builds a mapping between event data and tree/branch structures.
 * @param evnt Map of data pointers keyed by name
 * 
 * Creates internal mapping between the data containers and the logical
 * tree/branch/leaf structure used in physics analysis.
 */
void event_template::build_mapping(std::map<std::string, data_t*>* evnt){
    if (this -> tree_variable_link.size()){return;}

    std::map<std::string, data_t*>::iterator ite;
    std::map<std::string, std::string>::iterator itl;   
    std::vector<std::string> tr = this -> trees; 

    for (size_t x(0); x < tr.size(); ++x){
        this -> next_[tr[x]] = false; 
        for (ite = evnt -> begin(); ite != evnt -> end(); ++ite){
            bool s = tr[x] == ite -> second -> tree_name;
            if (!s){continue;}
            for (itl = this -> m_leaves.begin(); itl != this -> m_leaves.end(); ++itl){
                if (ite -> second -> leaf_name != itl -> second){continue;}
                std::vector<std::string> type_name = this -> split(itl -> first, "/"); 
                this -> tree_variable_link[tr[x]][type_name[0]].tree = tr[x]; 
                this -> tree_variable_link[tr[x]][type_name[0]].handle[type_name[1]] = ite -> second; 
                break; 
            } 
        }
    }
    this -> particle_link.clear(); 
}

/**
 * @brief Builds an event from the mapped data.
 * @param evnt Map of data pointers keyed by name
 * @return Map of event_template objects keyed by tree name
 * 
 * Constructs event_template objects for each tree in the mapping,
 * populating them with the corresponding data and particles.
 */
std::map<std::string, event_template*> event_template::build_event(std::map<std::string, data_t*>* evnt){
    this -> build_mapping(evnt); 
    std::map<std::string, event_template*> output = {}; 
    std::map<std::string, std::map<std::string, element_t>>::iterator itr = this -> tree_variable_link.begin();
    for (; itr != this -> tree_variable_link.end(); ++itr){
        if (this -> next_[itr -> first]){continue;}
        event_template* ev = this -> clone(); 
        ev -> tree = itr -> first; 

        std::map<std::string, element_t>::iterator itrx = this -> tree_variable_link[itr -> first].begin(); 
        for (; itrx != this -> tree_variable_link[itr -> first].end(); ++itrx){
            if (!itrx -> second.boundary()){delete ev; ev = nullptr; break;}
            itrx -> second.set_meta(); 

            if (itrx -> first == std::string(ev -> name)){
                ev -> build(&itrx -> second);
                ev -> index = itrx -> second.event_index; 
                ev -> hash = itrx -> second.filename; 
                ev -> filename = itrx -> second.filename;
            }
            else {
                std::map<std::string, particle_template*>* builder = new std::map<std::string, particle_template*>(); 
                ev -> particle_generators[itrx -> first] -> build(builder, &itrx -> second);
                std::map<std::string, particle_template*>* m_link = ev -> particle_link[itrx -> first]; 
                m_link -> insert(builder -> begin(), builder -> end()); 
                ev -> particle_link[itrx -> first] = builder; 
            }

            this -> next_[itr -> first] *= this -> tree_variable_link[itr -> first][itrx -> first].next(); 
        }
        if (!ev){continue;} 
        ev -> flush_leaf_string(); 
        output[itr -> first] = ev;
    }
    return output; 
}

/**
 * @brief Empty implementation of the build method (to be overridden in derived classes).
 * @param ele Pointer to an element_t structure
 * 
 * This is a placeholder method that derived classes are expected to override
 * with their specific event building logic.
 */
void event_template::build(element_t*){
    return; 
}

/**
 * @brief Clears all registered leaves, trees, and branches.
 * 
 * This method resets the internal data structures that maintain the
 * event's logical organization, including trees, branches, and leaves.
 * It also clears leaf data from all registered particles.
 */
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
    this -> deregister_particle(&this -> particle_generators); 
}

/**
 * @brief Clears all particles from the event.
 * 
 * This method removes all particles from the event, deregistering
 * them and freeing any associated memory.
 */
void event_template::flush_particles(){
    if (this -> next_.size()){return;}
    std::map<std::string, std::map<std::string, particle_template*>*>::iterator itr; 
    for (itr = this -> particle_link.begin(); itr != this -> particle_link.end(); ++itr){
        this -> deregister_particle(itr -> second); delete itr -> second;  itr -> second = nullptr; 
    }
    this -> particle_link = {}; 
}

/**
 * @brief Placeholder for event compilation logic.
 * 
 * This method is intended to be overridden in derived classes
 * to implement specific event compilation logic.
 */
void event_template::CompileEvent(){
    return; 
}

/**
 * @brief Creates a copy of the current event_template object.
 * @return Pointer to the newly created event_template object
 * 
 * This method creates a new instance of the event_template class,
 * copying the current object's state into the new instance.
 */
event_template* event_template::clone(){
    return new event_template(); 
}

