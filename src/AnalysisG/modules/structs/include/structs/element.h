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
enum class data_enum {vvf, vvd, vvl, vvi, vf, vl, vi, vc, vb, f, l, i, ull, b}; 

struct data_t {
    public:
        data_enum type; 

        std::vector<std::vector<std::vector<float>>>*  r_vvf = nullptr; 
        std::vector<std::vector<std::vector<double>>>* r_vvd = nullptr; 
        std::vector<std::vector<std::vector<long>>>*   r_vvl = nullptr; 
        std::vector<std::vector<std::vector<int>>>*    r_vvi = nullptr; 

        std::vector<std::vector<float>>* r_vf = nullptr; 
        std::vector<std::vector<long>>*  r_vl = nullptr; 
        std::vector<std::vector<int>>*   r_vi = nullptr; 
        std::vector<std::vector<char>>*  r_vc = nullptr; 
        std::vector<std::vector<bool>>*  r_vb = nullptr; 

        std::vector<float>* r_f = nullptr; 
        std::vector<long>*  r_l = nullptr; 
        std::vector<int>*   r_i = nullptr; 
        std::vector<bool>*  r_b = nullptr; 

        std::vector<unsigned long long>* r_ull = nullptr; 

        bool element(std::vector<std::vector<float>>* el);
        bool element(std::vector<std::vector<double>>* el);
        bool element(std::vector<std::vector<long>>* el);
        bool element(std::vector<std::vector<int>>* el);

        bool element(std::vector<float>* el); 
        bool element(std::vector<long>* el); 
        bool element(std::vector<int>* el); 
        bool element(std::vector<char>* el); 
        bool element(std::vector<bool>* el); 

        bool element(float* el);
        bool element(long* el);
        bool element(int* el); 
        bool element(bool* el); 
        bool element(unsigned long long* el); 


// ******************************************************************************************* //
        std::string   leaf_name = "";
        std::string branch_name = "";
        std::string   tree_name = ""; 
        std::string   leaf_type = ""; 
        std::string        path = ""; 

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

        template <typename T>
        bool next(T* el){
            bool sk = this -> file_index >= (int)this -> files_i -> size(); 
            if (sk){return false;}

            long idx = this -> files_i -> at(this -> file_index);
            if (this -> index < idx){
                this -> element(el); 
                this -> index++; 
                return true; 
            }

            this -> file_index++; 
            sk = this -> file_index >= (int)this -> files_i -> size(); 
            if (sk){return false;}
            this -> initialize();
            return this -> next(el); 
        }


    private:
        void flush_buffer(); 
        void fetch_buffer();

        void string_type();

        // ROOT IO functions
        template <typename T>
        bool flush_buffer(T** data){
            if (!(*data)){return false;}
            (*data) -> clear();
            delete *data; 
            *data = nullptr; 
            return true; 
        }; 

        template <typename T>
        void fetch_buffer(std::vector<T>** data){
            Long64_t l = this -> tree -> GetEntries(); 
            if (!*data){(*data) = new std::vector<T>();}
            (*data) -> reserve(l); 
            for (unsigned long x(0); x < l; ++x){
                this -> tree -> GetEntry(x); 
                (*data) -> push_back(*reinterpret_cast<T*>(this -> leaf -> GetValuePointer())); 
            }
        }; 
}; 

struct element_t {
    std::string tree = "";
    std::map<std::string, std::vector<std::vector<float>>>  r_vvf = {}; 
    std::map<std::string, std::vector<std::vector<double>>> r_vvd = {}; 
    std::map<std::string, std::vector<std::vector<long>>>   r_vvl = {}; 
    std::map<std::string, std::vector<std::vector<int>>>    r_vvi = {}; 
    
    std::map<std::string, std::vector<float>> r_vf = {}; 
    std::map<std::string, std::vector<long>>  r_vl = {}; 
    std::map<std::string, std::vector<int>>   r_vi = {}; 
    std::map<std::string, std::vector<char>>  r_vc = {}; 
    std::map<std::string, std::vector<bool>>  r_vb = {}; 
    
    std::map<std::string, float> r_f = {}; 
    std::map<std::string, long>  r_l = {}; 
    std::map<std::string, int>   r_i = {}; 
    std::map<std::string, bool>  r_b = {};  
    std::map<std::string, unsigned long long> r_ull = {};

    bool next(); 
    void set_meta(); 
    long event_index = -1; 
    std::string filename = ""; 

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
