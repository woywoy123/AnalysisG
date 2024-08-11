#ifndef IO_IO_H
#define IO_IO_H

#include <map>
#include <string>
#include <H5Cpp.h>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>

#include <TTreeReader.h>
#include <TTreeReaderArray.h>

#include <meta/meta.h>
#include <tools/tools.h>
#include <structs/folds.h>
#include <structs/element.h>
#include <notification/notification.h>

class io: 
    public tools, 
    public notification
{
    public:
        io(); 
        ~io(); 
       
        template <typename g>
        void write(std::vector<g>* inpt, std::string set_name){
            int length = inpt -> size(); 

            hid_t pairs = this -> member(g()); 
            H5::DataSet* dataset = this -> dataset(set_name, pairs, length); 
            if (!dataset){return;}

            dataset -> write(inpt -> data(), pairs); 
            hid_t id = this -> file -> getId(); 
            H5Fflush(id, H5F_SCOPE_LOCAL); 
        } 
 
        template <typename g>
        void write(g* inpt, std::string set_name){
            int length = 1; 
            hid_t pairs = this -> member(g()); 
            H5::DataSet* dataset = this -> dataset(set_name, pairs, length); 
            if (!dataset){return;}

            dataset -> write(inpt, pairs); 
            hid_t id = this -> file -> getId(); 
            H5Fflush(id, H5F_SCOPE_LOCAL); 
        }


        template <typename g>
        void read(std::vector<g>* outpt, std::string set_name){
            hid_t pairs = this -> member(g()); 
            H5::DataSet* dataset = this -> dataset(set_name);  
            if (!dataset){return;}
            H5::DataSpace space_r = dataset -> getSpace();
            hsize_t dim_r[1];
            space_r.getSimpleExtentDims(dim_r); 
            int length = dim_r[0];
            g* ptr = (g*)malloc(length * sizeof(g));
            dataset -> read(ptr, pairs); 
            for (int i(0); i < length; ++i){outpt -> push_back(ptr[i]);}
            free(ptr);
        } 

        template <typename g>
        void read(g* out, std::string set_name){
            hid_t pairs = this -> member(g());
            H5::DataSet* dataset = this -> dataset(set_name);
            if (!dataset){return;}
            H5::DataSpace space_r = dataset -> getSpace();
            hsize_t dim_r[1];
            space_r.getSimpleExtentDims(dim_r); 
            g* ptr = (g*)malloc(dim_r[0]*sizeof(g));
            dataset -> read(ptr, pairs); 
            *out = *ptr;
            free(ptr);
        }

        bool start(std::string filename, std::string read_write); 
        void end();
       
        std::vector<std::string> dataset_names(); 

        std::map<std::string, long> root_size(); 
        void check_root_file_paths(); 
        bool scan_keys(); 
        void root_begin(); 
        void root_end(); 

        std::map<std::string, data_t*>* get_data(); 

        bool enable_pyami = true; 
        std::string metacache_path = "./"; 
        std::string current_working_path = "."; 

        std::vector<std::string> trees = {}; 
        std::vector<std::string> branches = {}; 
        std::vector<std::string> leaves = {}; 

        std::map<std::string, TFile*> files_open = {}; 
        std::map<std::string, meta*>   meta_data = {}; 

        // key: Filename ; key tree_name : TTree*
        std::map<std::string, std::map<std::string, TTree*>> tree_data  = {}; 
        std::map<std::string, std::map<std::string, long>> tree_entries = {}; 

        // branch path : key branch_name : TBranch*
        std::map<std::string, std::map<std::string, TBranch*>> branch_data = {}; 

        // leaf filename : key leaf_name : TLeaf*
        std::map<std::string, std::map<std::string, TLeaf*>>      leaf_data = {}; 
        std::map<std::string, std::map<std::string, std::string>> leaf_typed = {}; 
        std::map<std::string, bool> root_files = {};

        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<std::string>>>> keys;

    private:
        hid_t member(folds_t t); 
        hid_t member(graph_hdf5_w t); 

        static herr_t file_info(hid_t loc_id, const char* name, const H5L_info_t* linfo, void *opdata); 

        std::map<std::string, H5::DataSet*> data_w; 
        std::map<std::string, H5::DataSet*> data_r; 
        H5::H5File* file = nullptr; 

        H5::DataSet* dataset(std::string set_name, hid_t type, int length); 
        H5::DataSet* dataset(std::string set_name); 

        TFile* file_root = nullptr; 
        void root_key_paths(std::string path); 
        void root_key_paths(std::string path, TTree* t); 
        void root_key_paths(std::string path, TBranch* t); 
        
        std::map<std::string, data_t*>*   iters = nullptr; 
        
        std::map<std::string, bool> missing_trigger = {}; 
        std::map<std::string, bool> success_trigger = {}; 

}; 

#endif
