/**
 * @file selection_template.h
 * @brief Defines the selection_template base class for event selection criteria.
 *
 * This file contains the declaration of the `selection_template` class, which serves
 * as the base class for implementing event selection criteria in the AnalysisG framework.
 * Selections filter events based on physics requirements and can output results to files.
 */

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

/**
 * @class selection_template
 * @brief Base class for implementing event selection criteria.
 *
 * The selection_template class provides the interface for defining event selections
 * that filter physics events based on specific criteria. Selections can be chained
 * together and their results can be written to output files.
 *
 * @section selection_usage Basic Usage
 *
 * Subclass selection_template to create custom selections:
 *
 * ```cpp
 * class DileptonSelection : public selection_template {
 * public:
 *     DileptonSelection() {
 *         name = "dilepton";
 *     }
 *     
 *     bool selection(event_template* ev) override {
 *         MyEvent* event = (MyEvent*)ev;
 *         int nLeptons = event->electrons.size() + event->muons.size();
 *         return nLeptons >= 2;
 *     }
 * };
 * ```
 *
 * @section selection_methods Key Methods
 *
 * - `selection()`: Override to implement selection logic (return true to pass)
 * - `strategy()`: Override for complex selection strategies
 * - `write()`: Output particle collections to files
 * - `merge()`: Combine results from multiple selections
 *
 * @section selection_helpers Helper Methods
 *
 * The class provides several template helper methods:
 * - `sum()`: Combine particles (four-momentum addition)
 * - `vectorize()`: Convert map to vector
 * - `make_unique()`: Remove duplicate particles
 * - `upcast()`/`downcast()`: Type conversion helpers
 * - `contains()`: Check if particle is in collection
 */
class selection_template: public tools
{
    public:
        /**
         * @brief Default constructor.
         * Initializes the selection template with default settings.
         */
        selection_template(); 
        
        /**
         * @brief Virtual destructor.
         * Cleans up resources including output handles.
         */
        virtual ~selection_template(); 

        cproperty<std::string, selection_template> name; ///< Selection name property.
        void static set_name(std::string*, selection_template*); ///< Setter for name.
        void static get_name(std::string*, selection_template*); ///< Getter for name.

        cproperty<std::string, selection_template> hash; ///< Unique hash identifier.
        void static set_hash(std::string*, selection_template*); ///< Setter for hash.
        void static get_hash(std::string*, selection_template*); ///< Getter for hash.

        cproperty<std::string, selection_template> tree; ///< Source tree name.
        void static get_tree(std::string*, selection_template*); ///< Getter for tree.

        cproperty<double, selection_template> weight; ///< Event weight.
        void static set_weight(double*, selection_template*); ///< Setter for weight.
        void static get_weight(double*, selection_template*); ///< Getter for weight.

        cproperty<long, selection_template> index; ///< Event index.
        void static set_index(long*, selection_template*); ///< Setter for index.
   
        /**
         * @brief Creates a clone of this selection.
         * @return Pointer to the cloned selection.
         */
        virtual selection_template* clone(); 
        
        /**
         * @brief The main selection method to override.
         * @param ev Pointer to the event to evaluate.
         * @return True if the event passes the selection, false otherwise.
         */
        virtual bool selection(event_template* ev);
        
        /**
         * @brief Alternative selection strategy method.
         * @param ev Pointer to the event to evaluate.
         * @return True if the event passes, false otherwise.
         */
        virtual bool strategy(event_template* ev);
        
        /**
         * @brief Merges results from another selection.
         * @param sel Pointer to the selection to merge from.
         */
        virtual void merge(selection_template* sel); 
        
        /**
         * @brief Bulk write operation for indices and hashes.
         * @param idx Pointer to the event index.
         * @param hx Pointer to the hash string.
         */
        virtual void bulk_write(const long* idx, std::string* hx); 
        
        /**
         * @brief Writes a particle collection to output.
         * @param particles Pointer to vector of particles to write.
         * @param name Name for this collection in output.
         * @param attrs Particle attributes to write.
         */
        virtual void write(std::vector<particle_template*>* particles, std::string name, particle_enum attrs); 

        /**
         * @brief Switch board for dispatching particle attributes.
         * @param attrs The attribute type to extract.
         * @param ptr Pointer to the particle.
         * @param data Output vector for 2D double data.
         */
        void switch_board(particle_enum attrs, particle_template* ptr, std::vector<std::vector<double>>* data); 
        
        /**
         * @brief Switch board overload for integer data.
         */
        void switch_board(particle_enum attrs, particle_template* ptr, std::vector<int>*    data); 
        
        /**
         * @brief Switch board overload for double data.
         */
        void switch_board(particle_enum attrs, particle_template* ptr, std::vector<double>* data); 
        
        /**
         * @brief Switch board overload for boolean data.
         */
        void switch_board(particle_enum attrs, particle_template* ptr, std::vector<bool>*   data); 

        /**
         * @brief Template method to write a variable to output.
         * @tparam g The type of variable to write.
         * @param var Pointer to the variable.
         * @param name Name for this variable in output.
         */
        template <typename g> 
        void write(g* var, std::string name){
            if (!this -> handle){return;}
            this -> handle -> process(&name) -> process(var, &name, this -> handle -> tree);
        }

        /**
         * @brief Template method to write a value to output.
         * @tparam g The type of value to write.
         * @param var The value to write.
         * @param name Name for this variable in output.
         */
        template <typename g> 
        void write(g var, std::string name){
            if (!this -> handle){return;}
            this -> handle -> process(&name) -> process(&var, &name, this -> handle -> tree);
        }

        /**
         * @brief Reverses hash strings to their original weight maps.
         * @param hashes Pointer to vector of hash strings.
         * @return Vector of weight maps.
         */
        std::vector<std::map<std::string, float>> reverse_hash(std::vector<std::string>* hashes); 

        /**
         * @brief Compiles the event for this selection.
         * @return True if compilation succeeded.
         */
        bool CompileEvent(); 
        
        /**
         * @brief Builds the selection from an event.
         * @param ev Pointer to the event.
         * @return Pointer to the built selection.
         */
        selection_template* build(event_template* ev); 
        
        /**
         * @brief Equality comparison operator.
         * @param p The selection to compare against.
         * @return True if selections are equal.
         */
        bool operator == (selection_template& p); 

        meta* meta_data = nullptr; ///< Pointer to metadata.
        std::string filename = ""; ///< Associated filename.
        event_t data; ///< Event data structure.

        /**
         * @brief Sums particles into a combined particle.
         * @tparam g Input particle type.
         * @tparam k Output particle type.
         * @param ch Pointer to vector of particles to sum.
         * @param out Pointer to output particle pointer.
         */
        template <typename g, typename k>
        void sum(std::vector<g*>* ch, k** out){
            k* prt = new k(); 
            prt -> _is_marked = true; 
            std::map<std::string, bool> maps; 
            for (size_t x(0); x < ch -> size(); ++x){
                if (maps[ch -> at(x) -> hash]){continue;}
                maps[ch -> at(x) -> hash] = true;
                prt -> iadd(ch -> at(x));
            }
            std::string hash_ = prt -> hash; 
            this -> garbage[hash_].push_back((particle_template*)prt); 
            (*out) = prt;  
        }

        /**
         * @brief Safely deletes particles not marked for preservation.
         * @tparam g The particle type.
         * @param particles Pointer to vector of particles.
         */
        template <typename g>
        void safe_delete(std::vector<g*>* particles){
            for (size_t x(0); x < particles -> size(); ++x){
                if (particles -> at(x) -> _is_marked){continue;}
                delete particles -> at(x); 
                (*particles)[x] = nullptr; 
            }
        }

        /**
         * @brief Sums particles from a map and returns the combined particle.
         * @tparam g The particle type.
         * @param ch Pointer to map of particles.
         * @return Pointer to the summed particle.
         */
        template <typename g>
        g* sum(std::map<std::string, g*>* ch){
            g* out = nullptr; 
            typename std::vector<g*> vec = this -> vectorize(ch); 
            this -> sum(&vec, &out); 
            return out; 
        }

        /**
         * @brief Sums particles and returns the invariant mass in GeV.
         * @tparam g The particle type.
         * @param ch Pointer to vector of particles.
         * @return Invariant mass of the combined system in GeV.
         */
        template <typename g>
        float sum(std::vector<g*>* ch){
            particle_template* prt = nullptr;
            this -> sum(ch, &prt); 
            return prt -> mass / 1000; 
        }

        /**
         * @brief Converts a map to a vector.
         * @tparam g The value type in the map.
         * @param in Pointer to the input map.
         * @return Vector of values from the map.
         */
        template <typename g>
        std::vector<g*> vectorize(std::map<std::string, g*>* in){
            typename std::vector<g*> out = {}; 
            typename std::map<std::string, g*>::iterator itr = in -> begin(); 
            for (; itr != in -> end(); ++itr){out.push_back(itr -> second);}
            return out; 
        }

        /**
         * @brief Removes duplicate particles based on hash.
         * @tparam g The particle type.
         * @param inpt Pointer to input vector.
         * @return Vector with duplicates removed.
         */
        template <typename g>
        std::vector<g*> make_unique(std::vector<g*>* inpt){
            std::map<std::string, g*> tmp; 
            for (size_t x(0); x < inpt -> size(); ++x){
                std::string hash = (*inpt)[x] -> hash; 
                tmp[hash] = (*inpt)[x]; 
            } 
   
            typename std::vector<g*> out = {}; 
            typename std::map<std::string, g*>::iterator itr; 
            for (itr = tmp.begin(); itr != tmp.end(); ++itr){out.push_back(itr -> second);}
            return out; 
        }

        /**
         * @brief Downcasts particles to base particle_template type.
         * @tparam g The derived particle type.
         * @param inpt Pointer to input vector of derived particles.
         * @param out Pointer to output vector of base particles.
         */
        template <typename g>
        void downcast(std::vector<g*>* inpt, std::vector<particle_template*>* out){
            for (size_t x(0); x < inpt -> size(); ++x){out -> push_back((particle_template*)(*inpt)[x]);}
        }

        /**
         * @brief Upcasts particles from a map to a derived type.
         * @tparam o The source particle type.
         * @tparam g The target particle type.
         * @param inpt Pointer to input map.
         * @param out Pointer to output vector.
         */
        template <typename o, typename g>
        void upcast(std::map<std::string, o*>* inpt, std::vector<g*>* out){
            typename std::map<std::string, o*>::iterator itx = inpt -> begin(); 
            for (; itx != inpt -> end(); ++itx){out -> push_back((g*)itx -> second);}
        }

        /**
         * @brief Upcasts particles from a vector to a derived type.
         * @tparam o The source particle type.
         * @tparam g The target particle type.
         * @param inpt Pointer to input vector.
         * @param out Pointer to output vector.
         */
        template <typename o, typename g>
        void upcast(std::vector<o*>* inpt, std::vector<g*>* out){
            for (size_t x(0); x < inpt -> size(); ++x){out -> push_back((g*)(*inpt)[x]);}
        }

        /**
         * @brief Extracts leptonic particles (leptons and neutrinos) from a map.
         * @tparam g The particle type.
         * @param inpt Input map of particles.
         * @param out Pointer to output vector.
         */
        template <typename g>
        void get_leptonics(std::map<std::string, g*> inpt, std::vector<particle_template*>* out){
            typename std::map<std::string, g*>::iterator itr = inpt.begin(); 
            for (; itr != inpt.end(); ++itr){
                if (!itr -> second -> is_lep && !itr -> second -> is_nu){continue;}
                out -> push_back((particle_template*)itr -> second);
            }
        }

        /**
         * @brief Checks if a particle is contained in a vector.
         * @tparam g The particle type in the vector.
         * @tparam j The particle type to check.
         * @param inpt Pointer to the vector to search.
         * @param pcheck Pointer to the particle to find.
         * @return True if the particle is found.
         */
        template <typename g, typename j>
        bool contains(std::vector<g*>* inpt, j* pcheck){
            for (size_t x(0); x < inpt -> size(); ++x){
                if ((*inpt)[x] -> hash != pcheck -> hash){continue;}
                return true;    
            }
            return false; 
        }

        int threadIdx = -1; ///< Thread index for parallel processing.
        std::map<std::string, std::map<std::string, float>> passed_weights = {}; ///< Weights for passed events.
        std::map<std::string, meta_t> matched_meta = {}; ///< Matched metadata.

    private:
        friend container;

        void bulk_write_out(); 
        void merger(selection_template* sl2); 

        std::unordered_map<long, std::string> sequence; 
        bool p_bulk_write = true; 
         
        write_t* handle = nullptr; 
        event_template* m_event = nullptr; 
        std::map<std::string, std::vector<particle_template*>> garbage = {}; ///< Garbage collection for created particles.
}; 


#endif
