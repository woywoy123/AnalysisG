#include <utils.h>
#include <pyc/pyc.h>

torch::Tensor utils::get_max(torch::Tensor inpt, int u){
    return std::get<u>(inpt.max({-1})).view({-1});
}

torch::Tensor utils::detach(torch::Tensor tn){
    return detach(&tn);
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

torch::Tensor utils::format(torch::Tensor* tn, long int l){
    return tn -> view({l}).contiguous();
}

torch::Tensor utils::format(torch::Tensor* tn, long int l1, long int l2){
    return tn -> view({l1, l2}).contiguous();
}

torch::Tensor utils::get_diff(torch::Tensor h1, torch::Tensor h2){
    return torch::cat({h1, h2 - h1}, {-1});
}

torch::Tensor utils::lzero(torch::Tensor* tn){
    return torch::zeros_like({*tn});
}

torch::Tensor utils::mzero(long int i, long int l){
    return torch::zeros({i, l});
}

torch::Tensor utils::mzero(long int i, long int l, bool form){
    return make_format(mzero(i, l));
}

bool utils::isnull(torch::Tensor inpt){
    return isnull(&inpt);
}

bool utils::isnull(torch::Tensor* inpt){
    long int lx = inpt -> view({-1}).size({0});
    if (!lx){return true;}
    return !inpt -> index({ inpt -> view({-1}) }).size({0});
}

torch::Tensor utils::get_index(torch::Tensor  h1, torch::Tensor msk){
    return h1.index({msk});
}

torch::Tensor utils::get_index(torch::Tensor* h1, torch::Tensor msk){
    return h1 -> index({msk});
}
