/**
 * @file properties.cxx
 * @brief Implementation of the event_template class methods.
 *
 * This file contains methods for managing event properties, including setting and retrieving
 * trees, branches, and metadata.
 */

#include <templates/event_template.h>

/**
 * @brief Sets the tree names for the event.
 *
 * @param inpt Vector of tree names.
 * @param ev Pointer to the event template.
 */
void event_template::set_trees(std::vector<std::string>* inpt, event_template* ev){
    for (size_t x(0); x < inpt -> size(); ++x){
        ev -> m_trees[inpt -> at(x)] = "tree"; 
    }
}

/**
 * @brief Sets the branch names for the event.
 *
 * @param inpt Vector of branch names.
 * @param ev Pointer to the event template.
 */
void event_template::set_branches(std::vector<std::string>* inpt, event_template* ev){
    for (size_t x(0); x < inpt -> size(); ++x){ev -> m_branches[inpt -> at(x)] = "branch";}
}

/**
 * @brief Adds a leaf to the event template.
 *
 * @param key The key name for the leaf.
 * @param val The value for the leaf (defaults to key if empty).
 */
void event_template::add_leaf(std::string key, std::string val){
    if (!val.size()){val = key;}
    std::string n = this -> name; 
    this -> m_leaves[n + "/" + key] = val; 
}

/**
 * @brief Gets all leaves from the event template.
 *
 * @param inpt Vector to store leaf names.
 * @param ev Pointer to the event template.
 */
void event_template::get_leaves(std::vector<std::string>* inpt, event_template* ev){
    std::map<std::string, std::string>::iterator itr = ev -> m_leaves.begin(); 
    for (; itr != ev -> m_leaves.end(); ++itr){inpt -> push_back(itr -> second);}
}

/**
 * @brief Sets the tree name for the event.
 *
 * @param name Pointer to the tree name.
 * @param ev Pointer to the event template.
 */
void event_template::set_tree(std::string* name, event_template* ev){
    ev -> data.name = *name; 
}

/**
 * @brief Gets the tree name from the event.
 *
 * @param name Pointer to store the tree name.
 * @param ev Pointer to the event template.
 */
void event_template::get_tree(std::string* name, event_template* ev){
    *name = ev -> data.name; 
}

/**
 * @brief Sets the name for the event.
 *
 * @param name Pointer to the name.
 * @param ev Pointer to the event template.
 */
void event_template::set_name(std::string* name, event_template* ev){
    ev -> data.name = *name; 
}

/**
 * @brief Sets the hash for the event.
 *
 * @param path Pointer to the path used for hashing.
 * @param ev Pointer to the event template.
 */
void event_template::set_hash(std::string* path, event_template* ev){
    if (ev -> data.hash.size()){return;}
    std::string x = *path + "/" + ev -> to_string(ev -> index); 
    ev -> data.hash = ev -> tools::hash(x, 18); 
    *path = ""; 
}

/**
 * @brief Gets the hash from the event.
 *
 * @param val Pointer to store the hash value.
 * @param ev Pointer to the event template.
 */
void event_template::get_hash(std::string* val, event_template* ev){
    *val = ev -> data.hash; 
}

/**
 * @brief Sets the weight for the event.
 *
 * @param inpt Pointer to the weight value.
 * @param ev Pointer to the event template.
 */
void event_template::set_weight(double* inpt, event_template* ev){
     ev -> data.weight = *inpt; 
}

/**
 * @brief Sets the index for the event.
 *
 * @param inpt Pointer to the index value.
 * @param ev Pointer to the event template.
 */
void event_template::set_index(long* inpt, event_template* ev){
    ev -> data.index = *inpt; 
}


