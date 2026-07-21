template <typename G, typename g>
bool AnalysisG::typecasting::as_vector(torch::Tensor* data, std::vector<G>* out, std::vector<signed long>* dims, g){
    torch::Tensor cpux = torch::empty(data -> sizes(), torch::device(torch::kCPU).pinned_memory(true).dtype(data -> dtype())); 
    if (!AnalysisG::typecasting::_transfer(data, &cpux)){return false;}
    cpux = cpux.reshape({-1}); 
    typename std::vector<g> linear(static_cast<g*>(cpux.data_ptr()), static_cast<g*>(cpux.data_ptr()) + cpux.numel()); 
    AnalysisG::typecasting::as_vector(out, &linear, dims, dims -> size()-1); 
    return true; 
}

template <typename g>
void AnalysisG::typecasting::as_vector(std::vector<g>* trgt, std::vector<g>* chnks, std::vector<signed long>*, int){
    trgt -> insert(trgt -> end(), chnks -> begin(), chnks -> end());  
}

template <typename G, typename g>
void AnalysisG::typecasting::as_vector(std::vector<G>* trgt, std::vector<g>* chnks, std::vector<signed long>* dims, int next_dim){
    std::vector<std::vector<g>> chnk_n = AnalysisG::tooling::discretize(chnks, (*dims)[next_dim]);
    for (size_t x(0); x < chnk_n.size(); ++x){
        G tmp = {}; 
        AnalysisG::typecasting::as_vector(&tmp, &chnk_n[x], dims, next_dim-1); 
        trgt -> push_back(tmp); 
    }
}

template <typename g>
void AnalysisG::typecasting::as_vector(torch::Tensor* data, std::vector<g>* out){
    std::vector<signed long> s = AnalysisG::typecasting::tensor_size(data); 
    AnalysisG::typecasting::as_vector(data, out, &s, g()); 
}


