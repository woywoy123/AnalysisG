#ifndef STRUCTS_ELEMENTS_H
#define STRUCTS_ELEMENTS_H

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>

#include <TTreeReader.h>
#include <TTreeReaderArray.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

// -------------------------- If you were directed here, simply add the data type within this section ----------------- //
enum class data_enum {
    vvf, vvd, vvl, vvi, vvb,
    vd, vf, vl, vi, vc, vb, 
    d, f, l, i, ull, b, ui              // <---- (1). add the type here v -- vector, vv -- vector<vector>, ....
}; 

struct data_t {
    public:
        data_enum type; 

        // ------------- (2). add the data type here -------------- //
        std::vector<std::vector<std::vector<float>>>*  r_vvf = nullptr; 
        std::vector<std::vector<std::vector<double>>>* r_vvd = nullptr; 
        std::vector<std::vector<std::vector<long>>>*   r_vvl = nullptr; 
        std::vector<std::vector<std::vector<int>>>*    r_vvi = nullptr; 
        std::vector<std::vector<std::vector<bool>>>*    r_vvb = nullptr; 

        std::vector<std::vector<long>>*   r_vl = nullptr; 
        std::vector<std::vector<double>>* r_vd = nullptr; 
        std::vector<std::vector<float>>*  r_vf = nullptr; 
        std::vector<std::vector<int>>*    r_vi = nullptr; 
        std::vector<std::vector<char>>*   r_vc = nullptr; 
        std::vector<std::vector<bool>>*   r_vb = nullptr; 

        std::vector<unsigned long long>* r_ull = nullptr; 
        std::vector<unsigned int>* r_ui = nullptr; 
        std::vector<double>* r_d = nullptr; 
        std::vector<long>*   r_l = nullptr; 
        std::vector<float>*  r_f = nullptr; 
        std::vector<int>*    r_i = nullptr; 
        std::vector<bool>*   r_b = nullptr; 
        
        // ------------- (3). add the data type interface -------------- //
        bool element(std::vector<std::vector<float>>* el);
        bool element(std::vector<std::vector<double>>* el);
        bool element(std::vector<std::vector<long>>* el);
        bool element(std::vector<std::vector<int>>* el);
        bool element(std::vector<std::vector<bool>>* el);

        bool element(std::vector<float>* el); 
        bool element(std::vector<double>* el); 
        bool element(std::vector<long>* el); 
        bool element(std::vector<int>* el); 
        bool element(std::vector<char>* el); 
        bool element(std::vector<bool>* el); 

        bool element(double* el); 
        bool element(float* el);
        bool element(long* el);
        bool element(int* el); 
        bool element(bool* el); 
        bool element(unsigned long long* el); 
        bool element(unsigned int* el); 

// -> go to modules/structs/cxx/structs.cxx
// ******************************************************************************************* //
//
        std::string   leaf_name = "";
        std::string branch_name = "";
        std::string   tree_name = ""; 
        std::string   leaf_type = ""; 
        std::string        path = ""; 
        std::string*      fname = nullptr; 

        TLeaf*     leaf = nullptr; 
        TBranch* branch = nullptr; 
        TTree*     tree = nullptr; 

        int file_index = 0;  
        long index = 0; 

        std::vector<std::string>* files_s = nullptr;
        std::vector<long>*        files_i = nullptr; 
        std::vector<TFile*>*      files_t = nullptr; 

        void initialize();
        void flush(); 
        bool next();

    private:
        void flush_buffer(); 
        void fetch_buffer();

        void string_type();

        // ROOT IO functions
        template <typename T>
        bool flush_buffer(T** data){
            if (!(*data)){return false;}
            delete *data; 
            *data = nullptr; 
            return true; 
        } 

        template <typename T>
        void fetch_buffer(std::vector<T>** data){
            TTreeReader r = TTreeReader(this -> tree); 
            TTreeReaderValue<T> dr(r, this -> branch_name.c_str()); 
            if (!*data){(*data) = new std::vector<T>();}
            while (r.Next()){(*data) -> push_back(*dr);}
        } 
}; 

struct element_t {
    std::string tree = "";

    bool next(); 
    void set_meta(); 
    long event_index = -1; 
    std::string filename = ""; 
    bool boundary(); 

    template <typename g>
    bool get(std::string key, g* var){
        if (!this -> handle.count(key)){return false;}
        if (this -> handle[key] -> element(var)){return true;}
        std::cout << "INVALID DATA TYPE GIVEN FOR: " + key << std::endl; 
        std::map<std::string, data_t*>::iterator itr = this -> handle.begin(); 
        for (; itr != this -> handle.end(); ++itr){
            data_t* d = itr -> second; 
            std::cout << "Leaf name: " << d -> leaf_name; 
            std::cout << "|" << d -> leaf_type << std::endl;
        }
        abort(); 
    }

    std::map<std::string, data_t*> handle = {}; 
}; 

#endif
