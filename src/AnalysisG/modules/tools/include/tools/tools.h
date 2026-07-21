#ifndef ANALYSISG_TOOLING_H
#define ANALYSISG_TOOLING_H

#include <iostream>
#include <cstdint>
#include <sstream>
#include <vector>
#include <string>
#include <map>

namespace AnalysisG::tooling {
    // io.cxx
    void create_path(std::string path); 
    void delete_path(std::string path); 
    bool is_file(std::string path); 
    void rename(std::string start, std::string target); 
    std::string absolute_path(std::string path); 
    std::vector<std::string> ls(std::string path, std::string ext = ""); 
    
    std::string to_string(double val, int prec); 
    std::string to_string(long double val, int prec); 
    
    void replace(std::string* in, std::string repl_str, std::string repl_with); 
    bool has_string(const std::string* inpt, std::string trg); 
    bool ends_with(const std::string* inpt, std::string val); 
    bool has_value(const std::vector<std::string>* data, std::string trg); 
    
    std::vector<std::string> split(std::string in, std::string delim);
    std::vector<std::string> split(std::string in, size_t n);
    std::string get_splits(const std::string* in, std::string delim, int index = -1); 
    
    std::string hash(std::string input, int len = 18);
    std::string lower(const std::string*); 
    
    std::string encode64(const std::string* data);
    std::string encode64(unsigned char const*, unsigned int len); 
    
    std::string decode64(std::string inpt);
    std::string decode64(const std::string* inpt); 
    
    // strings.cxx
    template <typename g>
    std::string to_string(g val); 

    template <typename g, typename k>
    g* as(k* ip); 

    template <typename G>
    std::vector<std::vector<G>> discretize(std::vector<G>* v, int N); 
    
    template <typename g>
    g max(std::vector<g>* inpt); 
    
    template <typename g>
    g min(std::vector<g>* inpt); 
   
    template <typename g>
    std::vector<g*> put(std::vector<g*>* src, std::vector<unsigned long>* trg); 
    
    template <typename g>
    void put(std::vector<g*>* out, std::vector<g*>* src, std::vector<unsigned long>* trg); 
    
    template <typename g>
    void unique_key(std::vector<g>* inx, std::vector<g>* oth); 
   
    template <typename g>
    bool pflush(g** p); 
    
    template <typename g>
    void vflush(std::vector<g*>* data, bool only_null = false); 
    
    template <typename k, typename g>
    void vflush(std::vector< std::map<k, g>* >* data); 
    
    template <typename k, typename g>
    void mflush(std::map<k, std::vector<g*>*>* data); 
    
    template <typename k, typename g>
    void mflush(std::map<k, g*>* data, bool only_null = false); 
    
    template <typename k, typename g>
    void mflush(std::map<k, std::vector<g>*>* data); 


    template <typename G>
    void merge(std::vector<G>* out, std::vector<G>* p2); 
    
    template <typename G>
    void merge(G* out, G* p2); 
    
    template <typename g, typename G>
    void merge(std::map<g, G>* out, std::map<g, G>* p2); 
  

    template <typename g>
    g sum(std::vector<g>* inpt); 
 
    template <typename G>
    void sum(G* out, G* p2); 
    
    template <typename G>
    void sum(std::vector<G>* out, std::vector<G>* p2); 
    
    template <typename g, typename G>
    void sum(std::map<g, G>* out, std::map<g, G>* p2); 
   

    int count(const std::string* str, const std::string sub); 

    template <typename g>
    void count(g* inp, long* ix); 
    
    template <typename g>
    void count(std::vector<g>* inp, long* ix);
   

    template <typename g>
    void contract(std::vector<g>* out, g* p2);
    
    template <typename g>
    void contract(std::vector<g>* out, std::vector<g>* p2); 
    
    template <typename g>
    void contract(std::vector<g>* out, std::vector<std::vector<g>>* p2); 
   

    template <typename g>
    void release_vector(std::vector<g>* ipt);  // needed for cython

    template <typename g>
    void nulls(g* d, int*); 

    template <typename g>
    void nulls(const std::vector<g>* d, int* mx_dim); 
   

    template <typename G, typename g>
    void as_primitive(G* data, std::vector<g>* lin, std::vector<signed long>*, unsigned int); 

    template <typename G, typename g>
    void as_primitive(std::vector<G>* data, std::vector<g>* linear, std::vector<signed long>* dims, unsigned int depth = 0); 


    template <typename g>
    void scout_dim(const g*, int*); 

    template <typename G>
    void scout_dim(const std::vector<G>* vec, int* mx_dim); 


    template <typename g>
    bool standard(const g*, int*); 
  
    template <typename g>
    bool standard(const std::vector<g>* vec, int* mx_dim); 
    
}

#include "merge.inl"
#include "memory.inl"
#include "tools.inl"
#include "primitives.inl"


#endif
