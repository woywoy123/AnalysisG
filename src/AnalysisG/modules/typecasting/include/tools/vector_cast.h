#ifndef TYPECASTING_VECTOR_CAST_H
#define TYPECASTING_VECTOR_CAST_H

#include <TInterpreter.h>
#include <torch/torch.h>
#include <vector>
#include <TTree.h>


template <typename G>
std::vector<std::vector<G>> chunking(std::vector<G>* v, int N){
    int n = v -> size(); 
    typename std::vector<std::vector<G>> out; 
    for (int ib = 0; ib < n; ib += N){
        int end = ib + N; 
        if (end > n){ end = n; }
        out.push_back(std::vector<G>(v -> begin() + ib, v -> begin() + end)); 
    }
    return out; 
}

template <typename g>
void tensor_vector(std::vector<g>* trgt, std::vector<g>* chnks, std::vector<signed long>* dims, int next_dim = 0){
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
void tensor_to_vector(torch::Tensor* data, std::vector<G>* out, std::vector<signed long>* dims, g prim){
    torch::Tensor tens = data -> reshape(-1).to(torch::kCPU, true); 
    torch::cuda::synchronize(); 
    typename std::vector<g> linear(tens.data_ptr<g>(), tens.data_ptr<g>() + tens.numel()); 
    tensor_vector(out, &linear, dims, dims -> size()-1); 
}



std::vector<signed long> tensor_size(torch::Tensor* inpt);
void add_to_dict(std::vector<std::vector<float>>* dummy); 
void add_to_dict(std::vector<std::vector<double>>* dummy); 
void add_to_dict(std::vector<std::vector<long>>* dummy); 
void add_to_dict(std::vector<std::vector<int>>* dummy); 
void add_to_dict(std::vector<std::vector<bool>>* dummy); 

void add_to_dict(std::vector<float>* dummy); 
void add_to_dict(std::vector<double>* dummy); 
void add_to_dict(std::vector<long>* dummy); 
void add_to_dict(std::vector<int>* dummy); 
void add_to_dict(std::vector<bool>* dummy); 

struct variable_t {
    public:
        void flush();
        void process(torch::Tensor* data, std::string* varname, TTree* tr);
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

        TBranch* tb = nullptr; 
        TTree*   tt = nullptr; 

        template <typename g, typename p>
        void add_data(std::vector<g>* type, torch::Tensor* data, std::vector<signed long>* s, std::string* varname, p prim){
            tensor_to_vector(data, type, s, prim); 
            if (this -> tb){return;}
            if (!this -> tt){return add_to_dict(type);}
            this -> tb = this -> tt -> Branch(varname -> c_str(), type); 
        }
};

#endif
