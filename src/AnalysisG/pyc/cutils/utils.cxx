#include <cutils/utils.h>

torch::Tensor clip(torch::Tensor* inpt, int dim){
    return inpt -> index({torch::indexing::Slice(), dim}); 
}

torch::Tensor format(std::vector<torch::Tensor>* inpt){
    std::vector<torch::Tensor> tmp; 
    for (unsigned int i(0); i < inpt -> size(); ++i){
        tmp.push_back(inpt -> at(i).view({-1, 1}));
    }
    return torch::cat(tmp, -1); 
}

torch::Tensor format(std::vector<torch::Tensor*> inpt){
    std::vector<torch::Tensor> tmp; 
    for (unsigned int i(0); i < inpt.size(); ++i){tmp.push_back(inpt[i] -> view({-1, 1}));}
    return torch::cat(tmp, -1); 
}

torch::TensorOptions MakeOp(torch::Tensor* x){
    return torch::TensorOptions().device(x -> device()).dtype(x -> dtype()); 
}

torch::Tensor changedev(std::string dev, torch::Tensor* inx){}
void changedev(torch::Tensor* inpt){}

