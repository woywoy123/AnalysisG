#include <pyc/cupyc.h>

torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor>* inpt){
    torch::Dict<std::string, torch::Tensor> out;  
    std::map<std::string, torch::Tensor>::iterator itr = inpt -> begin(); 
    for (; itr != inpt -> end(); ++itr){out.insert(itr -> first, itr -> second);}
    return out; 
}

torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor> inpt){
    return pyc::std_to_dict(&inpt); 
}

