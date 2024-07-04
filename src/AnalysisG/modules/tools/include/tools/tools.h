#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

#ifndef TOOLS_TOOLS_H
#define TOOLS_TOOLS_H

class tools
{
    public:
        // tools.cxx
        tools(); 
        ~tools(); 

        // io.cxx
        void create_path(std::string path); 
        void delete_path(std::string path); 
        bool is_file(std::string path); 
        std::string absolute_path(std::string path); 
        std::vector<std::string> ls(std::string path, std::string ext = ""); 

        // strings.cxx
        std::string to_string(double val); 
        void replace(std::string* in, std::string repl_str, std::string repl_with); 
        bool has_string(std::string* inpt, std::string trg); 
        bool ends_with(std::string* inpt, std::string val); 
        bool has_value(std::vector<std::string>* data, std::string trg); 

        std::vector<std::string> split(std::string in, std::string delim);
        std::vector<std::string> split(std::string in, int n);
        std::string hash(std::string input, int len = 18);
        std::string lower(std::string*); 
 
        // Template functions
        template <typename G>
        std::vector<std::vector<G>> discretize(std::vector<G>* v, int N){
            int n = v -> size(); 
            typename std::vector<std::vector<G>> out; 
            for (int ib = 0; ib < n; ib += N){
                int end = ib + N; 
                if (end > n){ end = n; }
                out.push_back(std::vector<G>(v -> begin() + ib, v -> begin() + end)); 
            }
            return out; 
        };

        template <typename g>
        g max(std::vector<g>* inpt){
            g ix = inpt -> at(0); 
            for (int t(1); t < inpt -> size(); ++t){
                if (inpt -> at(t) <= ix){continue;}
                ix = inpt -> at(t); 
            }
            return ix; 
        }

        template <typename g>
        g min(std::vector<g>* inpt){
            g ix = inpt -> at(0); 
            for (int t(1); t < inpt -> size(); ++t){
                if (inpt -> at(t) >= ix){continue;}
                ix = inpt -> at(t); 
            }
            return ix; 
        }
}; 


#endif
