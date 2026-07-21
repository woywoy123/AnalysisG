#ifndef EVENT_TEMPLATE_H
#define EVENT_TEMPLATE_H

#include <templates/particle_template.h>
#include <structs/property.h>
#include <structs/element.h>
#include <structs/event.h>
#include <tools/tools.h>
#include <meta/meta.h>

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
        void add_leaf(std::string key, std::string leaf = ""); 

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

        cproperty<long, event_template> index; 
        void static set_index(long*, event_template*); 

        std::map<std::string, std::string> m_trees; 
        std::map<std::string, std::string> m_branches;
        std::map<std::string, std::string> m_leaves; 
      
        virtual event_template* clone(); 
        virtual void build(element_t* el); 
        virtual void CompileEvent(); 

        std::vector<particle_template*> multi_neutrino(
                std::vector<particle_template*>* targets, double phi, double met, 
                double mt = 172.68 * 1000, double mw = 80.385 * 1000, 
                double violation = 1e-4, double limit = 0.1
        ); 

        std::map<std::string, event_template*> build_event(std::map<std::string, data_t*>* evnt); 


        template <typename m, typename g>
        std::vector<g*> vectorize(std::map<m, g*>* ipt){    
            typename std::vector<g*> vec; 
            typename std::map<m, g*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){vec.push_back(ix -> second);}
            return vec; 
        }

        template <typename m, typename g, typename k>
        void vectorize(std::map<m, g*>* ipt, std::vector<g*> opt){    
            typename std::map<m, g*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){opt -> push_back(ix -> second);}
        }

        template <typename g, typename m, typename k>
        std::vector<g*> vectorize(std::map<m, k*>* ipt){    
            typename std::vector<g*> vec; 
            typename std::map<m, k*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){vec.push_back(dynamic_cast<g*>(ix -> second));}
            return vec; 
        }

        template <typename g>
        std::map<int, g*> return_by_index(std::map<std::string, g*>* ipt){
            typename std::map<int, g*> xata = {}; 
            typename std::map<std::string, g*>::iterator ix = ipt -> begin();
            for (; ix != ipt -> end(); ++ix){xata[int(ix -> second -> index)] = ix -> second;}
            return xata; 
        }

        template <typename G>
        void register_particle(std::map<std::string, G*>* object){
            G* x = new G(); 
            std::string tp = x -> type; 

            std::map<std::string, std::string>::iterator itr = x -> leaves.begin(); 
            for (; itr != x -> leaves.end(); ++itr){
                this -> m_leaves[tp + "/" + itr -> first] = itr -> second;
            }
            x -> leaves.clear(); 
            this -> particle_link[tp] = (std::map<std::string, particle_template*>*)object; 
            this -> particle_generators[tp] = x; 
        }

        template <typename G>
        void deregister_particle(std::vector<G*>* object){
            if (!object -> size()){return;}
            for (size_t x(0); x < object -> size(); ++x){
                if (!object -> at(x)){continue;}
                std::string hash_ = object -> at(x) -> hash; 
                if (this -> garbage.count(hash_)){continue;}
                this -> garbage[hash_] = object -> at(x);
            }
        }

        template <typename g, typename k>
        g* sum(std::vector<k*>* ch){
            g* prt = new g(); 
            prt -> _is_marked = true; 
            std::map<std::string, bool> maps; 
            for (size_t x(0); x < ch -> size(); ++x){
                if (!ch -> at(x)){continue;}
                if (maps[ch -> at(x) -> hash]){continue;}
                maps[ch -> at(x) -> hash] = true;
                prt -> iadd(ch -> at(x));
                prt -> register_child(ch -> at(x)); 
            }
            std::string hash_ = prt -> hash; 
            if (this -> garbage.count(hash_)){this -> pflush(&prt);}
            else {this -> garbage[hash_] = prt;}
            return (prt) ? prt : dynamic_cast<g*>(this -> garbage[hash_]); 
        }


        template <typename g>
        g* sum(std::vector<g*>* ch){
            g* prt = new g(); 
            prt -> _is_marked = true; 
            std::map<std::string, bool> maps; 
            for (size_t x(0); x < ch -> size(); ++x){
                if (!ch -> at(x)){continue;}
                if (maps[ch -> at(x) -> hash]){continue;}
                maps[ch -> at(x) -> hash] = true;
                prt -> iadd(ch -> at(x));
                prt -> register_child(ch -> at(x)); 
            }
            std::string hash_ = prt -> hash; 
            if (this -> garbage.count(hash_)){this -> pflush(&prt);}
            else {this -> garbage[hash_] = prt;}
            return (prt) ? prt : dynamic_cast<g*>(this -> garbage[hash_]); 
        }


        template <typename k, typename g>
        g* sum(std::map<k, g*>* ch){
            typename std::vector<g*> vec = this -> vectorize(ch); 
            return this -> sum(&vec); 
        }

        bool operator == (event_template& p); 

        event_t data; 
        meta* meta_data = nullptr; 
        std::string filename = ""; 
        void flush_particles();

    private:
        void build_mapping(std::map<std::string, data_t*>* evnt); 
        std::map<std::string, particle_template*> garbage = {}; 
        void flush_leaf_string(); 

        template <typename G>
        void deregister_particle(std::map<std::string, G*>* object){
            if (!object -> size()){return;}
            typename std::map<std::string, G*>::iterator itr = object -> begin(); 
            for (; itr != object -> end(); ++itr){this -> pflush(&itr -> second);}
            object -> clear(); 
        }

        std::map<std::string, bool> next_ = {}; 
        std::map<std::string, particle_template*> particle_generators; 
        std::map<std::string, std::map<std::string, element_t>> tree_variable_link = {}; 
        std::map<std::string, std::map<std::string, particle_template*>*> particle_link = {}; 
}; 


#endif
