#ifndef EVENT_TEMPLATE_H
#define EVENT_TEMPLATE_H

#include <templates/particle_template.h>
#include <structs/property.h>
#include <structs/event.h>
#include <tools/tools.h>
#include <io/io.h>

#include <iostream>
#include <cstdint>
#include <string>
#include <vector>


class event_template: public tools
{
    public:
        event_template(); 
        virtual ~event_template(); 

        cproperty<std::vector<std::string>, event_template> trees;  
        void static set_trees(std::vector<std::string>*, event_template*); 


        cproperty<std::vector<std::string>, event_template> branches;  
        void static set_branches(std::vector<std::string>*, event_template*); 

        cproperty<std::vector<std::string>, event_template> leaves;  
        void static get_leaves(std::vector<std::string>*, event_template*); 
        void add_leaf(std::string key, std::string leaf); 

        cproperty<std::string, event_template> name; 
        void static set_name(std::string*, event_template*); 

        cproperty<std::string, event_template> hash; 
        void static set_hash(std::string*, event_template*); 
        void static get_hash(std::string*, event_template*); 

        cproperty<std::string, event_template> tree;  
        void static set_tree(std::string*, event_template*); 
        void static get_tree(std::string*, event_template*); 

        cproperty<double, event_template> weight;
        void static set_weight(double*, event_template*); 

        cproperty<double, event_template> index; 
        void static set_index(double*, event_template*); 

        std::map<std::string, std::string> m_trees; 
        std::map<std::string, std::string> m_branches;
        std::map<std::string, std::string> m_leaves; 
      
        virtual event_template* clone(); 
        virtual void build(element_t* el); 

        std::map<std::string, event_template*> build_event(std::map<std::string, data_t*> evnt); 

        template <typename G>
        void register_particle(std::map<std::string, G*>* object){
            G* x = new G(); 
            std::string tp = x -> type; 

            std::map<std::string, std::string>::iterator itr = x -> leaves.begin(); 
            for (; itr != x -> leaves.end(); ++itr){
                this -> m_leaves[tp + "/" + itr -> first] = itr -> second;
            }
            this -> particle_link[tp] = (std::map<std::string, particle_template*>*)object; 
            this -> particle_generators[tp] = x; 
        }; 

        bool operator == (event_template& p); 

        event_t data; 

    private:
        void build_mapping(std::map<std::string, data_t*> evnt); 

        std::map<std::string, event_template*> event_link = {}; 
        std::map<std::string, std::map<std::string, particle_template*>*> particle_link = {}; 
        std::map<std::string, particle_template*> particle_generators; 
        std::map<std::string, std::map<std::string, element_t>> tree_variable_link = {}; 
}; 


#endif
