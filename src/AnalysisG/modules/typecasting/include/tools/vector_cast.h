#ifndef TYPECASTING_VECTOR_CAST_H
#define TYPECASTING_VECTOR_CAST_H

#ifdef PYC_CUDA
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#endif

#include <tools/tools.h>
#include <structs/meta.h>
#include <structs/base.h>
#include <torch/torch.h>
#include <TTree.h>
#include <vector>

struct write_t; 

struct variable_t: public bsc_t 
{
    public:
        variable_t(); 
        variable_t(bool use_external); 
        ~variable_t() override; 

        void create_meta(meta_t* mt);
        void build_switch(size_t s, torch::Tensor* tx); 

        void process(torch::Tensor* data, std::string* varname, TTree* tr);

        // =========================== Add your type (3) =========================== //
        void process(std::vector<std::vector<float>>*  data, std::string* varname, TTree* tr); 
        void process(std::vector<std::vector<double>>* data, std::string* varname, TTree* tr); 
        void process(std::vector<std::vector<long>>*   data, std::string* varname, TTree* tr); 
        void process(std::vector<std::vector<int>>*    data, std::string* varname, TTree* tr); 
        void process(std::vector<std::vector<bool>>*   data, std::string* varname, TTree* tr); 

        void process(std::vector<float>*  data, std::string* varname, TTree* tr); 
        void process(std::vector<double>* data, std::string* varname, TTree* tr); 
        void process(std::vector<long>*   data, std::string* varname, TTree* tr); 
        void process(std::vector<int>*    data, std::string* varname, TTree* tr); 
        void process(std::vector<bool>*   data, std::string* varname, TTree* tr); 

        void process(float*  data, std::string* varname, TTree* tr); 
        void process(double* data, std::string* varname, TTree* tr); 
        void process(long*   data, std::string* varname, TTree* tr); 
        void process(int*    data, std::string* varname, TTree* tr); 
        void process(bool*   data, std::string* varname, TTree* tr); 
        // ========================================================================= //

        std::string variable_name = ""; 
        bool failed_branch = false; 

    private: 
        friend write_t;
        bool use_external = false; 
        bool is_triggered = false; 

        TBranch* tb = nullptr; 
        TTree*   tt = nullptr; 
        meta_t* mtx = nullptr; 

        template <typename g, typename p>
        void add_data(g*& tx, torch::Tensor* data, std::vector<signed long>* s, p prim){
            if (!tx){tx = new g();}
            tensor_to_vector(data, tx, s, prim); 
            if (this -> tb || !this -> tt){return;}
            this -> tb = this -> tt -> Branch(this -> variable_name.c_str(), tx); 
            this -> failed_branch = !this -> tb; 
        }
       
        template <typename g>
        void add_data(g* var, g*& tx, std::string* name, TTree* tr = nullptr){
            if (!tx){this -> variable_name = *name; tx = new g();}
            if (!var){return;}
            *tx = *var; 
            if (this -> tb || !this -> tt){return;}
            if (tr){this -> tt = tr;}
            this -> tb = this -> tt -> Branch(this -> variable_name.c_str(), tx); 
            this -> failed_branch = !this -> tb; 
            this -> is_triggered = true; 
        }
};

#endif
