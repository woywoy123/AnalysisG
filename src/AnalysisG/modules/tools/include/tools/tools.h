#ifndef TOOLS_TOOLS_H
#define TOOLS_TOOLS_H

#include <iostream>
#include <cstdint>
#include <sstream>
#include <vector>
#include <string>
#include <map>

class tools
{
    public:
        tools();
        ~tools();  

        // io.cxx
        static void create_path(std::string path); 
        static void delete_path(std::string path); 
        static bool is_file(std::string path); 
        static void rename(std::string start, std::string target); 
        static std::string absolute_path(std::string path); 
        static std::vector<std::string> ls(std::string path, std::string ext = ""); 

        // strings.cxx
        template <typename g>
        static std::string to_string(g val){
            std::stringstream ss; 
            ss << val; 
            return ss.str(); 
        } 

        static std::string to_string(double val, int prec); 
        static std::string to_string(long double val, int prec); 

        static void replace(std::string* in, std::string repl_str, std::string repl_with); 
        static bool has_string(const std::string* inpt, std::string trg); 
        static bool ends_with(const std::string* inpt, std::string val); 
        static bool has_value(const std::vector<std::string>* data, std::string trg); 

        static std::vector<std::string> split(std::string in, std::string delim);
        static std::vector<std::string> split(std::string in, size_t n);
        static std::string get_splits(std::string* in, std::string delim, int index = -1); 

        static std::string hash(std::string input, int len = 18);
        static std::string lower(std::string*); 

        static std::string encode64(std::string* data);
        static std::string encode64(unsigned char const*, unsigned int len); 

        static std::string decode64(std::string* inpt);
        static std::string decode64(std::string const& s); 

        template <typename g>
        g* as(){return (g*)this;}

        // template functions
        template <typename G>
        static std::vector<std::vector<G>> discretize(std::vector<G>* v, int N){
            size_t n = v -> size(); 
            typename std::vector<std::vector<G>> out; 
            out.reserve(int(v -> size()/N)); 
            for (size_t ib = 0; ib < n; ib += N){
                size_t end = ib + N; 
                if (end > n){ end = n; }
                out.push_back(std::vector<G>(v -> begin() + ib, v -> begin() + end)); 
            }
            return out; 
        }

        template <typename g>
        static g max(std::vector<g>* inpt){
            g ix = inpt -> at(0); 
            for (size_t t(1); t < inpt -> size(); ++t){
                if (inpt -> at(t) <= ix){continue;}
                ix = inpt -> at(t); 
            }
            return ix; 
        }

        template <typename g>
        static g min(std::vector<g>* inpt){
            g ix = inpt -> at(0); 
            for (size_t t(1); t < inpt -> size(); ++t){
                if (inpt -> at(t) >= ix){continue;}
                ix = inpt -> at(t); 
            }
            return ix; 
        }

        template <typename g>
        static g sum(std::vector<g>* inpt){
            g ix = 0; 
            for (size_t t(0); t < inpt -> size(); ++t){ix += (*inpt)[t];}
            return ix; 
        }

        template <typename g>
        static std::vector<g*> put(std::vector<g*>* src, std::vector<unsigned long>* trg){
            typename std::vector<g*> out(src -> size(), nullptr); 
            for (size_t x(0); x < trg -> size(); ++x){out[x] = (*src)[(*trg)[x]];}
            return out; 
        }

        template <typename g>
        static void put(std::vector<g*>* out, std::vector<g*>* src, std::vector<unsigned long>* trg){
            out -> clear(); 
            out -> reserve(trg -> size());
            for (size_t x(0); x < trg -> size(); ++x){
                g* v = (*src)[(*trg)[x]];  
                out -> push_back(v);
                v -> in_use = 1; 
            }
        }

        template <typename g>
        static void unique_key(std::vector<g>* inx, std::vector<g>* oth){
            typename std::map<g, bool> ch;
            for (size_t x(0); x < oth -> size(); ++x){ch[(*oth)[x]] = true;}
            for (size_t x(0); x < inx -> size(); ++x){
                g kx = (*inx)[x];
                if (ch[kx]){continue;}
                oth -> push_back(kx);
                ch[kx] = true;
            }
        }

        template <typename g>
        static bool pflush(g** p){
            if (!(*p)){return true;}
            delete *p; 
            (*p) = nullptr; 
            return false; 
        }
        
        template <typename g>
        static void vflush(std::vector<g*>* data){
            if (!data){return;}
            for (size_t x(0); x < data -> size(); ++x){tools::pflush(&(*data)[x]);}
            data -> clear(); 
            data -> shrink_to_fit(); 
        }

        template <typename k, typename g>
        static void vflush(std::vector< std::map<k, g>* >* data){
            if (!data){return;}
            for (size_t x(0); x < data -> size(); ++x){tools::pflush(&(*data)[x]);}
            data -> clear(); 
            data -> shrink_to_fit(); 
        }

        template <typename k, typename g>
        static void mflush(std::map<k, std::vector<g*>*>* data){
            if (!data){return;}
            typename std::map<k, std::vector<g*>*>::iterator ix; 
            for (ix = data -> begin(); ix != data -> end(); ++ix){
                tools::vflush(ix -> second); 
                tools::pflush(&ix -> second); 
            }
            data -> clear(); 
        }

        template <typename k, typename g>
        static void mflush(std::map<k, g*>* data){
            if (!data){return;}
            typename std::map<k, g*>::iterator ix; 
            for (ix = data -> begin(); ix != data -> end(); ++ix){
                if (!ix -> second){continue;}
                delete ix -> second; 
            }
            data -> clear(); 
        }


        template <typename k, typename g>
        static void mflush(std::map<k, std::vector<g>*>* data){
            if (!data){return;}
            typename std::map< k, std::vector<g>* >::iterator ix; 
            for (ix = data -> begin(); ix != data -> end(); ++ix){
                if (!ix -> second){continue;}
                delete (*data)[ix -> first]; 
                (*data)[ix -> first] = nullptr; 
            }
            data -> clear(); 
        }


}; 


#endif
