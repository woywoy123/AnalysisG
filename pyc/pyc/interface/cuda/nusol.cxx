#include <pyc/cupyc.h>
#include <nusol/nusol.cuh>
#include <utils/utils.cuh>

torch::Dict<std::string, torch::Tensor> pyc::nusol::BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses){
    changedev(&pmc_b); 
    return pyc::std_to_dict(nusol_::BaseMatrix(&pmc_b, &pmc_mu, &masses)); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy, 
        torch::Tensor masses, torch::Tensor sigma, double null
){
    changedev(&pmc_b); 
    std::map<std::string, torch::Tensor> out = nusol_::Nu(&pmc_b, &pmc_mu, &met_xy, &masses, &sigma, null);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
        torch::Tensor met_xy, torch::Tensor masses, double null
){
    changedev(&pmc_b1); 
    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&pmc_b1, &pmc_b2, &pmc_l1, &pmc_l2, &met_xy, null, &masses);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::combinatorial(
        torch::Tensor edge_index, torch::Tensor batch, torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy, 
        double mT, double mW, double top_pm, double w_pm, long steps, double null, bool gev
){
    changedev(&edge_index);
    std::map<std::string, torch::Tensor> out;
    out = nusol_::combinatorial(&edge_index, &batch, &pmc, &pid, &met_xy, mT, mW, top_pm, w_pm, steps, null, gev); 
    return pyc::std_to_dict(&out); 
}


