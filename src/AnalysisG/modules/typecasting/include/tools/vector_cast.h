#ifndef TYPECASTING_VECTOR_CAST_H
#define TYPECASTING_VECTOR_CAST_H

#include <torch/torch.h>
#include <vector>

std::vector<signed long> tensor_size(torch::Tensor* inpt){
    c10::IntArrayRef dims = inpt -> sizes();
    std::vector<signed long> out; 
    for (size_t x(0); x < dims.size(); ++x){out.push_back(dims[x]);}
    return out;  
}

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
    torch::Tensor tens = data -> view({-1}).to(torch::kCPU); 
    typename std::vector<g> linear(tens.data_ptr<g>(), tens.data_ptr<g>() + tens.numel()); 
    tensor_vector(out, &linear, dims, dims -> size()-1); 
}


#endif
