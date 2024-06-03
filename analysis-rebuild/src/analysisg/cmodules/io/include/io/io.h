#ifndef IO_IO_H
#define IO_IO_H

#include <map>
#include <H5Cpp.h>
#include <tools/tools.h>
#include <structs/particles.h>

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

        // ROOT IO functions
        void check_root_file_paths(); 
        void scan_keys(); 

        bool enable_pyami = true; 
        std::string metacache_path = "./"; 

        std::vector<std::string> trees = {}; 
        std::vector<std::string> branches = {}; 
        std::vector<std::string> leaves = {}; 
        std::map<std::string, bool> root_files = {};

        std::string current_working_path = "."; 
    private:
        H5::CompType member(particle_t t); 

        static herr_t file_info(hid_t loc_id, const char* name, const H5L_info_t* linfo, void *opdata); 

        std::map<std::string, H5::DataSet*> data_w; 
        std::map<std::string, H5::DataSet*> data_r; 
        H5::H5File* file = nullptr; 

        H5::DataSet* dataset(std::string set_name, H5::CompType type, int length); 
        H5::DataSet* dataset_in_group(H5::Group& group, std::string set_name, H5::CompType type, int length); 
        H5::DataSet* dataset(std::string set_name); 

        std::vector<std::string> root_key_paths(); 

}; 

#endif
