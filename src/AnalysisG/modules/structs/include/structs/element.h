#ifndef STRUCTS_ELEMENTS_H
#define STRUCTS_ELEMENTS_H

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>

#include <TTreeReaderArray.h>
#include <TTreeReader.h>

#include <tools/vector_cast.h>
#include <structs/base.h>
#include <structs/meta.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>

struct data_t: public bsc_t 
{
    public:
        data_t(); 
        ~data_t() override;

        std::string   leaf_name = "";
        std::string branch_name = "";
        std::string   tree_name = ""; 
        std::string   leaf_type = ""; 
        std::string        path = ""; 
        std::string*      fname = nullptr; 

        TLeaf*     leaf = nullptr; 
        TBranch* branch = nullptr; 
        TTree*     tree = nullptr; 
        TFile*     file = nullptr; 

        int file_index = 0;  

        std::vector<std::string>* files_s = nullptr;
        std::vector<long>*        files_i = nullptr; 
        std::vector<TFile*>*      files_t = nullptr; 

        void initialize();
        void flush(); 
        bool next();

    private:
        void fetch_buffer();
        void string_type();

        template <typename T>
        void fetch_buffer(std::vector<T>** data){
            TTreeReader r = TTreeReader(this -> tree); 
            TTreeReaderValue<T> dr(r, this -> branch_name.c_str()); 
            if (*data){(*data) -> clear();}
            if (!(*data)){(*data) = new std::vector<T>();}
            while (r.Next()){(*data) -> push_back(*dr);}
            (*data) -> shrink_to_fit(); 
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
        std::map<std::string, data_t*>::iterator itr; 
        for (itr = this -> handle.begin(); itr != this -> handle.end(); ++itr){
            data_t* d = itr -> second; 
            std::cout << "Leaf name: " << d -> leaf_name; 
            std::cout << "|" << d -> leaf_type << std::endl;
        }
        abort(); 
    }
    std::map<std::string, data_t*> handle = {}; 
}; 


struct write_t {
    TFile* file = nullptr; 
    TTree* tree = nullptr;
    meta_t* mtx = nullptr; 
    std::map<std::string, variable_t*>* data = nullptr; 

    variable_t* process(std::string* name);
    void write(); 
    void create(std::string tr_name, std::string path); 
    void close(); 
}; 


struct writer {
    public:
        writer(); 
        ~writer();
        void create(std::string* pth); 
        void write(std::string* tree);

        template <typename g>
        void process(std::string* tree, std::string* name, g* t){
            this -> process(tree, name) -> process(t, name, nullptr);
        }

    private:
        write_t* head = nullptr; 
        std::map<std::string, write_t*> handle = {};  
        variable_t* process(std::string* tree, std::string* name); 

}; 



#endif
