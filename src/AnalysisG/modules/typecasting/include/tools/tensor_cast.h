#ifndef TYPECASTING_TENSOR_CAST_H
#define TYPECASTING_TENSOR_CAST_H

#include <vector>
#include <torch/torch.h>

// --------- tensor padding --------- //
template <typename g>
void scout_dim(g*, int*){return;}

template <typename g>
void nulls(g* d, int*){*d = -1;}

template <typename g>
bool standard(g*, int*){ return true; }

template <typename G, typename g>
void as_primitive(G* data, std::vector<g>* lin, std::vector<signed long>*, unsigned int){lin -> push_back(*data);} 

template <typename G>
void scout_dim(const std::vector<G>* vec, int* mx_dim){
    int dim_ = 0;
    for (size_t x(0); x < vec -> size(); ++x){
        scout_dim(&vec -> at(x), &dim_);
        if (!dim_){dim_ = vec -> size();}
    }
    if (dim_ < *mx_dim){return;}
    *mx_dim = dim_; 
}

template <typename g>
void nulls(const std::vector<g>* d, int* mx_dim){
    for (size_t t(d -> size()); t < *mx_dim; ++t){
        d -> push_back({});
        nulls(&d -> at(t), mx_dim);
    }
} 

template <typename g>
bool standard(const std::vector<g>* vec, int* mx_dim){
    size_t l = vec -> size();
    if (!l){nulls(vec, mx_dim);}
    for (size_t x(0); x < l; ++x){
        if (!standard(&vec -> at(x), mx_dim)){continue;}
        nulls(vec, mx_dim);
        return false;
    };
    return false; 
}

template <typename G, typename g>
static void as_primitive(std::vector<G>* data, std::vector<g>* linear, std::vector<signed long>* dims, unsigned int depth = 0){
    if (depth == dims -> size()){dims -> push_back(data -> size());}
    for (size_t x(0); x < data -> size(); ++x){
        G tx = (*data)[x]; 
        as_primitive(&tx, linear, dims, depth+1);
    }
} 


template <typename G, typename g>
static torch::Tensor build_tensor(std::vector<G>* _data, at::ScalarType _op, g, torch::TensorOptions* op){
    int max_dim = 0; 
    std::vector<g> linear = {};
    std::vector<signed long> dims = {}; 

    scout_dim(_data, &max_dim); 
    standard(_data, &max_dim);
    as_primitive(_data, &linear, &dims); 

    size_t s = linear.size(); 
    g* d = new g[s]; 
    for (size_t x(0); x < s; ++x){d[x] = linear[x];}
    if (dims.size() == 1){dims.push_back(1);}
    torch::Tensor ten = torch::from_blob(d, dims, (*op).dtype(_op)).clone(); 
    delete [] d;  
    return ten; //torch::from_blob((void*)linear.data(), dims, (*op).dtype(_op)).clone(); 
}

#endif
