#include <cutils/utils.cuh>

const dim3 BLOCKS(unsigned int threads, unsigned int dx){
    const dim3 blocks( (dx + threads -1) / threads); 
    return blocks; 
}


const dim3 BLOCKS(unsigned int threads, unsigned int dx, unsigned int dy){
    const dim3 blocks( (dx + threads -1) / threads, dy); 
    return blocks; 
}

const dim3 BLOCKS(unsigned int threads, unsigned int len, unsigned int dy, unsigned int dz){
    const dim3 blocks( (len + threads -1) / threads, dy, dz); 
    return blocks; 
}

torch::TensorOptions MakeOp(torch::Tensor* v){
    return torch::TensorOptions().dtype(v -> scalar_type()).device(v -> device()); 
}

torch::Tensor format(std::vector<torch::Tensor> v, std::vector<signed long> dim){
    std::vector<torch::Tensor> out; 
    for (size_t x(0); x < v.size(); ++x){out.push_back(v[x].view(dim));}
    return torch::cat(out, -1); 
}
