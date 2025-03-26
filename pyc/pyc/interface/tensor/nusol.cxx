#include <pyc/tpyc.h>
#include <nusol/nusol.h>

torch::Tensor pyc::nusol::BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses){
    return nusol_::Hperp(&pmc_b, &pmc_mu, &masses); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy, 
        torch::Tensor masses, torch::Tensor sigma, double null
){
    std::map<std::string, torch::Tensor> out = nusol_::Nu(&pmc_b, &pmc_mu, &met_xy, &masses, &sigma, null);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
        torch::Tensor met_xy, torch::Tensor masses, double null
){
    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&pmc_b1, &pmc_b2, &pmc_l1, &pmc_l2, &met_xy, null, &masses);
    return pyc::std_to_dict(&out); 
}

TORCH_LIBRARY(tpyc, m){
    m.def("nusol_base_basematrix", &pyc::nusol::BaseMatrix); 
    m.def("nusol_nu"             , &pyc::nusol::Nu); 
    m.def("nusol_nunu"           , &pyc::nusol::NuNu); 
}
