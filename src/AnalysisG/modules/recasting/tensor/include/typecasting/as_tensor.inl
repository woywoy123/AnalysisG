template <typename G, typename g>
torch::Tensor AnalysisG::typecasting::as_tensor(std::vector<G>* _data, at::ScalarType _op, g, torch::TensorOptions* op){
    int max_dim = 0; 
    std::vector<g> linear = {};
    std::vector<signed long> dims = {}; 

    AnalysisG::tooling::scout_dim(_data, &max_dim); 
    AnalysisG::tooling::standard(_data, &max_dim);
    AnalysisG::tooling::as_primitive(_data, &linear, &dims); 

    size_t s = linear.size(); 
    g* d = new g[s]; 
    for (size_t x(0); x < s; ++x){d[x] = linear[x];}
    if (dims.size() == 1){dims.push_back(1);}
    torch::Tensor ten = torch::from_blob(d, dims, (*op).dtype(_op)).clone(); 
    delete [] d;  
    return ten; //torch::from_blob((void*)linear.data(), dims, (*op).dtype(_op)).clone(); 
}


