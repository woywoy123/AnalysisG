#ifndef IO_IO_H
#define IO_IO_H

#include <map>
#include <H5Cpp.h>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>

#include <tools/tools.h>
#include <structs/particles.h>

enum class data_enum {vvf, vvl, vvi, vf, vl, vi, f, l, i}; 

struct data_t {
    public:
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
        data_enum type; 

        std::vector<std::vector<std::vector<float>>>* r_vvf = nullptr; 
        std::vector<std::vector<std::vector<long>>>*  r_vvl = nullptr; 
        std::vector<std::vector<std::vector<int>>>*   r_vvi = nullptr; 

        std::vector<std::vector<float>>* r_vf = nullptr; 
        std::vector<std::vector<long>>*  r_vl = nullptr; 
        std::vector<std::vector<int>>*   r_vi = nullptr; 

        std::vector<float>* r_f = nullptr; 
        std::vector<long>*  r_l = nullptr; 
        std::vector<int>*   r_i = nullptr; 

        std::vector<std::string>* files_s = nullptr;
        std::vector<long>*        files_i = nullptr; 
        std::vector<TFile*>*      files_t = nullptr; 

        void initialize();
        void flush(); 

        template <typename T>
        bool next(T* el){
            std::cout << "-> " << this -> index << std::endl;
            std::cout << "-> " << this -> file_index << std::endl;
            if (this -> file_index >= this -> files_i -> size()){return false;}

            long idx = this -> files_i -> at(this -> file_index); 
            if (this -> index >= idx){
                this -> file_index++; 
                TFile* tf = this -> files_t -> at(this -> file_index); 
                this -> tree = tf -> Get<TTree>(this -> tree_name.c_str()); 
                this -> branch = this -> tree -> FindBranch(this -> branch_name.c_str()); 
                this -> leaf   = this -> tree -> FindLeaf(this -> path.c_str()); 
                this -> flush_buffer(); 
                this -> fetch_buffer(); 
                this -> index = 0;
            } 
            this -> element(el); 
            this -> index++; 
            return true; 
        }

    private:
        void flush_buffer(); 
        void fetch_buffer();

        void string_type();

        void element(std::vector<std::vector<float>>* el);
        void element(std::vector<std::vector<long>>* el);
        void element(std::vector<std::vector<int>>* el);

        void element(float* el);

        // ROOT IO functions
        template <typename T>
        bool flush_buffer(T** data){
            if (!*data){return false;}
            (*data) -> clear();
            delete *data; 
            *data = nullptr; 
            return true; 
        }; 

        template <typename T>
        void fetch_buffer(std::vector<T>** data){
            TTreeReader r = TTreeReader(this -> tree); 
            TTreeReaderValue<T> dr(r, this -> branch_name.c_str()); 
            if (!*data){(*data) = new std::vector<T>();}
            while (r.Next()){(*data) -> push_back(*dr);}
        }; 

}; 


class io: public tools
{
    public:
        io(); 
        ~io(); 
       
        template <typename g>
        void write(std::map<std::string, g>* inpt, std::string set_name){
            int length = inpt -> size(); 
           
            H5::CompType pairs = this -> member(g()); 
            H5::DataSet* dataset = this -> dataset(set_name, pairs, length); 
            if (!dataset){return;}

            typename std::vector<g> writer; 
            typename std::map<std::string, g>::iterator itr = inpt -> begin();
            for (; itr != inpt -> end(); ++itr){writer.push_back(itr -> second);} 
            dataset -> write(writer.data(), pairs); 
            hid_t id = this -> file -> getId(); 
            H5Fflush(id, H5F_SCOPE_LOCAL); 
        }; 

        template <typename g>
        void write(std::vector<g>* inpt, std::string set_name){
            int length = inpt -> size(); 

            H5::CompType pairs = this -> member(g()); 
            H5::DataSet* dataset = this -> dataset(set_name, pairs, length); 
            if (!dataset){return;}

            dataset -> write(inpt -> data(), pairs); 
            hid_t id = this -> file -> getId(); 
            H5Fflush(id, H5F_SCOPE_LOCAL); 
        }; 
 
        template <typename g>
        void write(g* inpt, std::string set_name){
            int length = 1; 
            H5::CompType pairs = this -> member(g()); 
            H5::DataSet* dataset = this -> dataset(set_name, pairs, length); 
            if (!dataset){return;}

            dataset -> write(inpt, pairs); 
            hid_t id = this -> file -> getId(); 
            H5Fflush(id, H5F_SCOPE_LOCAL); 
        }

        template <typename g>
        void read(std::map<std::string, g>* outpt, std::string set_name){
            H5::CompType pairs = this -> member(g()); 
            H5::DataSet* dataset = this -> dataset(set_name);  
            if (!dataset){return;}
            H5::DataSpace space_r = dataset -> getSpace();
            hsize_t dim_r[1];
            space_r.getSimpleExtentDims(dim_r); 
            int length = dim_r[0];
            g* ptr = (g*)malloc(length * sizeof(g));
            dataset -> read(ptr, pairs); 
            for (int i(0); i < length; ++i){(*outpt)[ptr[i].key()] = ptr[i];}
            free(ptr);
        }; 

        template <typename g>
        void read(g* out, std::string set_name){
            H5::CompType pairs = this -> member(g());
            H5::DataSet* dataset = this -> dataset(set_name);
            if (!dataset){return;}
            H5::DataSpace space_r = dataset -> getSpace();
            hsize_t dim_r[1];
            space_r.getSimpleExtentDims(dim_r); 
            g* ptr = (g*)malloc(sizeof(g));
            dataset -> read(ptr, pairs); 
            *out = *ptr;
            free(ptr);
       };

        void write(std::map<std::string, particle_t>* inpt, std::string set_name); 
        bool start(std::string filename, std::string read_write); 
        void end();
       
        std::vector<std::string> dataset_names(); 
        bool has_dataset_name(std::string name); 

        std::map<std::string, long> root_size(); 
        void check_root_file_paths(); 
        void scan_keys(); 
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
        H5::CompType member(particle_t t); 

        static herr_t file_info(hid_t loc_id, const char* name, const H5L_info_t* linfo, void *opdata); 

        std::map<std::string, H5::DataSet*> data_w; 
        std::map<std::string, H5::DataSet*> data_r; 
        H5::H5File* file = nullptr; 

        H5::DataSet* dataset(std::string set_name, H5::CompType type, int length); 
        H5::DataSet* dataset_in_group(H5::Group& group, std::string set_name, H5::CompType type, int length); 
        H5::DataSet* dataset(std::string set_name); 

        TFile* file_root = nullptr; 
        void root_key_paths(std::string path); 
        void root_key_paths(std::string path, TTree* t); 
        void root_key_paths(std::string path, TBranch* t); 
        
        std::map<std::string, data_t*>* iters = nullptr; 

}; 

#endif
