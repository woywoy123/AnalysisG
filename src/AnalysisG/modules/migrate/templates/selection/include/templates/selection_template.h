#ifndef SELECTION_TEMPLATE_H
#define SELECTION_TEMPLATE_H

#include <templates/particle_template.h>
#include <templates/event_template.h>

#include <structs/property.h>
#include <structs/event.h>
#include <structs/enums.h>
#include <meta/meta.h>

#include <tools/vector_cast.h>
#include <tools/merge_cast.h>
#include <tools/tools.h>

class container; 

class selection_template: public tools
{
    public:
        selection_template(); 
        virtual ~selection_template(); 

        cproperty<std::string, selection_template> name; 
        void static set_name(std::string*, selection_template*); 
        void static get_name(std::string*, selection_template*);

        cproperty<std::string, selection_template> hash; 
        void static set_hash(std::string*, selection_template*); 
        void static get_hash(std::string*, selection_template*); 

        cproperty<std::string, selection_template> tree;  
        void static get_tree(std::string*, selection_template*); 

        cproperty<double, selection_template> weight;
        void static set_weight(double*, selection_template*); 
        void static get_weight(double*, selection_template*); 

        cproperty<long, selection_template> index; 
        void static set_index(long*, selection_template*); 
   
        virtual selection_template* clone(); 
        virtual bool selection(event_template* ev);
        virtual bool strategy(event_template* ev);
        virtual void merge(selection_template* sel); 
        virtual void bulk_write(const long* idx, std::string* hx); 
        virtual void write(std::vector<particle_template*>* particles, std::string _name, particle_enum attrs); 

        template <typename g>
        void write(std::vector<g*>* particles, std::string _name, particle_enum attrs){
            std::vector<particle_template*> ox; 
            this -> downcast(particles, &ox); 
            this -> write(&ox, _name, attrs); 
        }

        void switch_board(particle_enum attrs, particle_template* ptr, std::vector<std::vector<double>>* data); 
        void switch_board(particle_enum attrs, particle_template* ptr, std::vector<int>*    data); 
        void switch_board(particle_enum attrs, particle_template* ptr, std::vector<double>* data); 
        void switch_board(particle_enum attrs, particle_template* ptr, std::vector<bool>*   data); 

        template <typename g> 
        void write(g* var, std::string _name){
            if (!this -> handle){return;}
            this -> handle -> process(&_name) -> process(var, &_name, this -> handle -> tree);
        }

        template <typename g> 
        void write(g var, std::string _name){
            if (!this -> handle){return;}
            this -> handle -> process(&_name) -> process(&var, &_name, this -> handle -> tree);
        }

        std::vector<std::map<std::string, float>> reverse_hash(std::vector<std::string>* hashes); 

        bool CompileEvent(); 
        selection_template* build(event_template* ev); 
        bool operator == (selection_template& p); 

        meta* meta_data = nullptr; 
        std::string filename = ""; 
        event_t data; 

        template <typename g, typename k>
        void sum(std::vector<g*>* ch, k** out){
            k* prt = new k(); 
            prt -> _is_marked = true; 
            std::map<std::string, bool> maps; 
            for (size_t x(0); x < ch -> size(); ++x){
                if (!ch -> at(x)){continue;}
                if (maps[ch -> at(x) -> hash]){continue;}
                maps[ch -> at(x) -> hash] = true;
                prt -> iadd(ch -> at(x));
            }
            std::string hash_ = prt -> hash; 
            this -> garbage[hash_].push_back((particle_template*)prt); 
            (*out) = prt;  
        }

        template <typename g>
        void safe_delete(std::vector<g*>* particles){
            for (size_t x(0); x < particles -> size(); ++x){
                if (particles -> at(x) -> _is_marked){continue;}
                delete particles -> at(x); 
                (*particles)[x] = nullptr; 
            }
        }

        template <typename g>
        g* sum(std::map<std::string, g*>* ch){
            g* out = nullptr; 
            typename std::vector<g*> vec = this -> vectorize(ch); 
            this -> sum(&vec, &out); 
            return out; 
        }

        template <typename g>
        float sum(std::vector<g*>* ch){
            particle_template* prt = nullptr;
            this -> sum(ch, &prt); 
            return prt -> mass / 1000; 
        }

        template <typename g>
        std::vector<g*> vectorize(std::map<std::string, g*>* in){
            typename std::vector<g*> out = {}; 
            typename std::map<std::string, g*>::iterator itr = in -> begin(); 
            for (; itr != in -> end(); ++itr){out.push_back(itr -> second);}
            return out; 
        }

        template <typename g>
        std::vector<g*> make_unique(std::vector<g*>* inpt){
            std::map<std::string, g*> tmp; 
            for (size_t x(0); x < inpt -> size(); ++x){
                std::string hash_ = (*inpt)[x] -> hash; 
                tmp[hash_] = (*inpt)[x]; 
            } 
   
            typename std::vector<g*> out = {}; 
            typename std::map<std::string, g*>::iterator itr; 
            for (itr = tmp.begin(); itr != tmp.end(); ++itr){out.push_back(itr -> second);}
            return out; 
        }

        template <typename g>
        void downcast(std::vector<g*>* inpt, std::vector<particle_template*>* out){
            for (size_t x(0); x < inpt -> size(); ++x){out -> push_back((particle_template*)(*inpt)[x]);}
        }

        template <typename o, typename g>
        void upcast(std::map<std::string, o*>* inpt, std::vector<g*>* out){
            typename std::map<std::string, o*>::iterator itx = inpt -> begin(); 
            for (; itx != inpt -> end(); ++itx){out -> push_back((g*)itx -> second);}
        }

        template <typename o, typename g>
        void upcast(std::vector<o*>* inpt, std::vector<g*>* out){
            for (size_t x(0); x < inpt -> size(); ++x){out -> push_back((g*)(*inpt)[x]);}
        }

        template <typename g>
        void get_leptonics(std::map<std::string, g*> inpt, std::vector<particle_template*>* out){
            typename std::map<std::string, g*>::iterator itr = inpt.begin(); 
            for (; itr != inpt.end(); ++itr){
                if (!itr -> second -> is_lep && !itr -> second -> is_nu){continue;}
                out -> push_back((particle_template*)itr -> second);
            }
        }

        template <typename g, typename j>
        bool contains(std::vector<g*>* inpt, j* pcheck){
            for (size_t x(0); x < inpt -> size(); ++x){
                if ((*inpt)[x] -> hash != pcheck -> hash){continue;}
                return true;    
            }
            return false; 
        }

        int threadIdx = -1; 
        std::map<std::string, std::map<std::string, float>> passed_weights = {}; 
        std::map<std::string, meta_t> matched_meta = {}; 

    private:
        friend container;

        void bulk_write_out(); 
        void merger(selection_template* sl2); 

        std::unordered_map<long, std::string> sequence; 
        bool p_bulk_write = true; 
         
        write_t* handle = nullptr; 
        event_template* m_event = nullptr; 
        std::map<std::string, std::vector<particle_template*>> garbage = {}; 
}; 


#endif
