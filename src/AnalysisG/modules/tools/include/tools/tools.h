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
        std::string to_string(double val, int prec); 
        void replace(std::string* in, std::string repl_str, std::string repl_with); 
        bool has_string(std::string* inpt, std::string trg); 
        bool ends_with(std::string* inpt, std::string val); 
        bool has_value(std::vector<std::string>* data, std::string trg); 

        std::vector<std::string> split(std::string in, std::string delim);
        std::vector<std::string> split(std::string in, int n);
        std::string hash(std::string input, int len = 18);
        std::string lower(std::string*); 

        std::string encode64(std::string* data);
        std::string encode64(unsigned char const*, unsigned int len); 

        std::string decode64(std::string* inpt);
        std::string decode64(std::string const& s); 

        // template functions
        template <typename G>
        std::vector<std::vector<G>> discretize(std::vector<G>* v, int N){
            int n = v -> size(); 
            typename std::vector<std::vector<G>> out; 
            out.reserve(int(v -> size()/N)); 
            for (int ib = 0; ib < n; ib += N){
                int end = ib + N; 
                if (end > n){ end = n; }
                out.push_back(std::vector<G>(v -> begin() + ib, v -> begin() + end)); 
            }
            return std::move(out); 
        }

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

        template <typename g>
        g sum(std::vector<g>* inpt){
            g ix = 0; 
            for (int t(0); t < inpt -> size(); ++t){ix += (*inpt)[t];}
            return ix; 
        }

        template <typename g>
        std::vector<g*> put(std::vector<g*>* src, std::vector<int>* trg){
            typename std::vector<g*> out(src -> size(), nullptr); 
            for (size_t x(0); x < trg -> size(); ++x){out[x] = (*src)[(*trg)[x]];}
            return std::move(out); 
        }

        template <typename g>
        void put(std::vector<g*>* out, std::vector<g*>* src, std::vector<int>* trg){
            out -> clear(); 
            for (size_t x(0); x < trg -> size(); ++x){
                g* v = (*src)[(*trg)[x]];  
                out -> push_back(v);
                v -> in_use = 1; 
            }
        }
}; 


#endif
