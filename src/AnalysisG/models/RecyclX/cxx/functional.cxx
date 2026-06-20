#include <utils.h>
#include <pyc/pyc.h>

torch::Tensor utils::detach(torch::Tensor tn){
    return utils::detach(&tn);
}

torch::Tensor utils::detach(torch::Tensor* tn){
    return tn -> detach();
}

torch::Tensor utils::as_type(torch::Tensor* tn, torch::ScalarType vs){
    return tn -> to(vs);
}

torch::Tensor utils::as_l(torch::Tensor* tn){
    return as_type(tn, torch::kLong);
}

torch::Tensor utils::as_f(torch::Tensor* tn){
    return as_type(tn, torch::kFloat32);
}

torch::Tensor utils::get_max(torch::Tensor inpt){
    return std::get<1>(inpt.max({-1})).view({-1});
}

torch::Tensor utils::get_diff(torch::Tensor h1, torch::Tensor h2){
    return torch::cat({h1, h2 - h1}, {-1});
}

torch::Tensor utils::get_index(torch::Tensor  h1, torch::Tensor msk){
    return h1.index({msk}).contiguous();
}

torch::Tensor utils::get_index(torch::Tensor* h1, torch::Tensor msk){
    return h1 -> index({msk}).contiguous();
}

torch::Tensor utils::get_index(torch::Tensor* h1, long int l){
    torch::Tensor x = h1 -> index({l}); 
    return utils::format(&x, -1); 
}

torch::Tensor utils::format(torch::Tensor* tn, long int l){
    return tn -> view({l}).contiguous();
}

torch::Tensor utils::format(torch::Tensor* tn, long int l1, long int l2){
    return tn -> view({l1, l2}).contiguous();
}


torch::Tensor utils::lzero(torch::Tensor* tn){
    return torch::zeros_like({*tn});
}

torch::Tensor utils::lzero(torch::Tensor* tn, torch::Tensor ix){
    return torch::zeros_like({tn -> index({ ix.view({-1}) })});
}

torch::Tensor utils::mzero(long int i, long int l){
    return torch::zeros({i, l}).to(torch::kFloat32);
}

torch::Tensor utils::mzero(long int i, long int l, bool form){
    return utils::mzero(i, l);
}

bool utils::isnull(torch::Tensor inpt){
    return utils::isnull(&inpt);
}

bool utils::isnull(torch::Tensor* inpt){
    long int lx = inpt -> view({-1}).size({0});
    if (!lx){return true;}
    return !inpt -> index({ inpt -> view({-1}) }).size({0});
}

torch::Tensor utils::node_idx(torch::Tensor* batch_index){
    torch::Tensor one = torch::ones_like(*batch_index);
    one = one.cumsum({0})-1;
    return one.to(torch::kLong);
}



