#include "../abstractions/cytypes.h"
#include "../code/code.h"
#include <sstream>

#ifndef ABSTRACTION_H
#define ABSTRACTION_H

namespace Tools
{
    std::string Hashing(std::string input); 
    std::string ToString(double input);
    std::vector<std::string> split(std::string inpt, std::string search); 
    std::string join(std::vector<std::string>* inpt, int s, int e, std::string delim); 
    int count(std::string inpt, std::string search); 
    std::vector<std::vector<std::string>> Quantize(const std::vector<std::string>& v, int N); 
}

namespace Abstraction
{
    class CyBase
    {
        public:
            CyBase();
            ~CyBase();

            void Hash(std::string inpt); 
            
            std::string hash = "";
            int event_index = 0; 
    }; 

    class CyEvent
    {
        public: 
            CyEvent(); 
            ~CyEvent();

            void ImportMetaData(meta_t meta); 
            void add_eventname(std::string event); 
            std::string Hash();

            meta_t  meta; 
            event_t event;
            graph_t graph; 
            selection_t selection;

            bool is_event = false; 
            bool is_graph = false; 
            bool is_selection = false; 

            template <typename T, typename G> 
            void set_event_hash(T* type, G* event){
                type -> event_hash = event -> event_hash; 
            };

            template <typename T, typename G>
            void set_event_tag(T* type, G* event){
                type -> event_tagging = event -> event_tagging;
            };

            template <typename T, typename G>
            void set_event_tree(T* type, G* event){
                type -> event_tree = event -> event_tree; 
            };

            template <typename T, typename G>
            void set_event_root(T* type, G* event){
                type -> event_root = event -> event_root; 
            };

            template <typename T, typename G>
            void set_event_index(T* type, G* event){
                type -> event_index = event -> event_index;
            }; 

            template <typename T, typename G>
            bool is_same(T* event1, G* event2)
            {
                if (event1 -> event_hash != event2 -> event_hash){ 
                    return false; 
                }
                if (event1 -> event_name != event2 -> event_name){ 
                    return false; 
                }
                if (event1 -> event_tree != event2 -> event_tree){
                    return false; 
                }
                if (event1 -> event_tagging != event2 -> event_tagging){
                    return false; 
                }
                if (event1 -> event != event2 -> event){
                    return false;
                }
                if (event1 -> graph != event2 -> graph){
                    return false;
                }
                if (event1 -> selection != event2 -> selection){
                    return false;
                }
                return true; 
            }; 


            Code::CyCode* code_link = nullptr; 


    };
}
#endif
