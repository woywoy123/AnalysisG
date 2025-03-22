#ifndef TYPECASTING_VECTOR_CAST_H
#define TYPECASTING_VECTOR_CAST_H

#include <TInterpreter.h>
#include <TSystem.h>
#include <TTree.h>
#include <tools/cfg.h>
#include <torch/torch.h>
#include <vector>

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
void tensor_to_vector(torch::Tensor* data, std::vector<G>* out, std::vector<signed long>* dims, g){
    torch::Tensor tens = data -> reshape(-1).to(torch::kCPU); 
    torch::cuda::synchronize(); 
    typename std::vector<g> linear(tens.data_ptr<g>(), tens.data_ptr<g>() + tens.numel()); 
    tensor_vector(out, &linear, dims, dims -> size()-1); 
}

std::vector<signed long> tensor_size(torch::Tensor* inpt);
void add_to_dict(std::vector<std::vector<double>>* dummy); 
void add_to_dict(std::vector<std::vector<float>>* dummy); 
void add_to_dict(std::vector<std::vector<long>>* dummy); 
void add_to_dict(std::vector<std::vector<int>>* dummy); 
void add_to_dict(std::vector<std::vector<bool>>* dummy); 

void add_to_dict(std::vector<double>* dummy); 
void add_to_dict(std::vector<float>* dummy); 
void add_to_dict(std::vector<long>* dummy); 
void add_to_dict(std::vector<int>* dummy); 
void add_to_dict(std::vector<bool>* dummy); 

void add_to_dict(double* dummy); 
void add_to_dict(float* dummy); 
void add_to_dict(long* dummy); 
void add_to_dict(int* dummy); 
void add_to_dict(bool* dummy); 

template <typename g>
void tensor_to_vector(torch::Tensor* data, std::vector<g>* out){
    std::vector<signed long> s = tensor_size(data); 
    tensor_to_vector(data, out, &s, g()); 
}


struct variable_t {
    public:

        void flush();
        void process(torch::Tensor* data, std::string* varname, TTree* tr);

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
        std::string variable_name = ""; 

    private: 
        std::vector<std::vector<float>>  vvf = {}; 
        std::vector<std::vector<double>> vvd = {}; 
        std::vector<std::vector<long>>   vvl = {}; 
        std::vector<std::vector<int>>    vvi = {}; 
        std::vector<std::vector<bool>>   vvb = {}; 

        std::vector<float>               vf = {}; 
        std::vector<double>              vd = {}; 
        std::vector<long>                vl = {}; 
        std::vector<int>                 vi = {}; 
        std::vector<bool>                vb = {}; 

        float   f = 0; 
        double  d = 0; 
        long    l = 0; 
        int     i = 0; 
        bool    b = 0; 

        TBranch* tb = nullptr; 
        TTree*   tt = nullptr; 

        template <typename g, typename p>
        void add_data(std::vector<g>* type, torch::Tensor* data, std::vector<signed long>* s, std::string* varname, p prim){
            tensor_to_vector(data, type, s, prim); 
            if (this -> tb){return;}
            if (!this -> tt){return add_to_dict(type);}
            this -> tb = this -> tt -> Branch(varname -> c_str(), type); 
            this -> tt -> AddBranchToCache(this -> tb, true); 
        }

        template <typename g>
        void add_data(g* var, g* type){
            *type = *var; 
            if (this -> tb){return;}
            this -> tb = this -> tt -> Branch(this -> variable_name.c_str(), type); 
        }
        
        template <typename g>
        void add_data(g* var, g* type, std::string* name, TTree* tr){
            if (this -> tt){return this -> add_data(var, type);}
            add_to_dict(var);
            this -> tt = tr; 
            this -> variable_name = *name;
            this -> add_data(var, type); 
        }
};

#endif
