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

            std::map<std::string, Code::CyCode*> this_code = {}; 
    };
}
#endif
