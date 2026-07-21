#include <typecasting/detensor.h>
#include <torch/torch.h>

bool AnalysisG::typecasting::_transfer(torch::Tensor* data, torch::Tensor* cpux){
//    c10::cuda::CUDAStream strx = at::cuda::getStreamFromPool(false, data -> device().index()); 
//    at::cuda::setCurrentCUDAStream(strx); 
//    AT_CUDA_CHECK(cudaStreamSynchronize(strx)); 
    cpux -> copy_(*data, true);
    torch::cuda::synchronize(data -> device().index()); 
    if (!cpux -> is_pinned()){return false;}
    return true; 
}

std::vector<signed long> AnalysisG::typecasting::tensor_size(torch::Tensor* inpt){
    c10::IntArrayRef dims = inpt -> sizes();
    std::vector<signed long> out; 
    for (size_t x(0); x < dims.size(); ++x){out.push_back(dims[x]);}
    return out;  
}



