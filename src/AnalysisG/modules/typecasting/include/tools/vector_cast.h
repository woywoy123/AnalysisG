#ifndef TYPECASTING_VECTOR_CAST_H
#define TYPECASTING_VECTOR_CAST_H

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <structs/meta.h>
#include <structs/base.h>
#include <torch/torch.h>
#include <TTree.h>
#include <vector>

bool _transfer(torch::Tensor* data, torch::Tensor* cpux); 

template <typename G>
std::vector<std::vector<G>> chunking(std::vector<G>* v, int N){
    size_t n = v -> size(); 
    typename std::vector<std::vector<G>> out; 
    for (size_t ib = 0; ib < n; ib += N){
        size_t end = ib + N; 
        if (end > n){ end = n; }
        out.push_back(std::vector<G>(v -> begin() + ib, v -> begin() + end)); 
    }
    return out; 
}

template <typename g>
void tensor_vector(std::vector<g>* trgt, std::vector<g>* chnks, std::vector<signed long>*, int){
    trgt -> insert(trgt -> end(), chnks -> begin(), chnks -> end());  
}


template <typename G, typename g>
void tensor_vector(std::vector<G>* trgt, std::vector<g>* chnks, std::vector<signed long>* dims, int next_dim = 0){
    std::vector<std::vector<g>> chnk_n = chunking(chnks, (*dims)[next_dim]);
    for (size_t x(0); x < chnk_n.size(); ++x){
        G tmp = {}; 
        tensor_vector(&tmp, &chnk_n[x], dims, next_dim-1); 
        trgt -> push_back(tmp); 
    }
}


template <typename G, typename g>
bool tensor_to_vector(torch::Tensor* data, std::vector<G>* out, std::vector<signed long>* dims, g){
    torch::Tensor cpux = torch::empty(data -> sizes(), torch::device(torch::kCPU).pinned_memory(true).dtype(data -> dtype())); 
    if (!_transfer(data, &cpux)){return false;}
    cpux = cpux.reshape({-1}); 
    typename std::vector<g> linear(static_cast<g*>(cpux.data_ptr()), static_cast<g*>(cpux.data_ptr()) + cpux.numel()); 
    tensor_vector(out, &linear, dims, dims -> size()-1); 
    return true; 
}

std::vector<signed long> tensor_size(torch::Tensor* inpt);

template <typename g>
void tensor_to_vector(torch::Tensor* data, std::vector<g>* out){
    std::vector<signed long> s = tensor_size(data); 
    tensor_to_vector(data, out, &s, g()); 
}

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
