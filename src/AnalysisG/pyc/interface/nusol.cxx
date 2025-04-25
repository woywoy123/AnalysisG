#include <pyc/pyc.h>

#ifdef PYC_CUDA
#include <utils/utils.cuh>
#include <nusol/nusol.cuh>
#else 
#include <utils/utils.h>
#include <nusol/nusol.h>
#endif

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
        torch::Tensor met_xy, torch::Tensor masses, double null, 
        const double step, const double tolerance, const unsigned int timeout
){
    changedev(&pmc_b1); 
    std::map<std::string, torch::Tensor> out;
    out = nusol_::NuNu(&pmc_b1, &pmc_b2, &pmc_l1, &pmc_l2, &met_xy, null, &masses, nullptr, step, tolerance, timeout);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
        torch::Tensor met_xy, double null, torch::Tensor mass1, torch::Tensor mass2, 
        const double step, const double tolerance, const unsigned int timeout
){
    changedev(&pmc_b1); 
    std::map<std::string, torch::Tensor> out; 
    out = nusol_::NuNu(&pmc_b1, &pmc_b2, &pmc_l1, &pmc_l2, &met_xy, null, &mass1, &mass2, step, tolerance, timeout);
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::combinatorial(
        torch::Tensor edge_index, torch::Tensor batch, torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy, 
        double mT, double mW, double null, double perturb, long steps, bool gev
){
    changedev(&edge_index);
    std::map<std::string, torch::Tensor> out;
    out = nusol_::combinatorial(&edge_index, &batch, &pmc, &pid, &met_xy, mT, mW, null, perturb, steps, gev); 
    return pyc::std_to_dict(&out); 
}

std::vector<std::pair<neutrino*, neutrino*>> pyc::nusol::combinatorial(
       std::vector<double> met_, std::vector<double> phi_, std::vector<std::vector<particle_template*>> particles,
       std::string dev, double mT, double mW, double null, double perturb, long steps
){

    std::vector<std::vector<particle_template*>> quarks, leptons; 
    for (size_t x(0); x < particles.size(); ++x){
        quarks.push_back({}); leptons.push_back({}); 
        for (size_t y(0); y < particles[x].size(); ++y){
            bool is_b = particles[x][y] -> is_b;
            if (is_b){quarks[x].push_back(particles[x][y]); continue;}

            bool is_l = particles[x][y] -> is_lep; 
            if (is_l){leptons[x].push_back(particles[x][y]); continue;}
        }
    }
    std::vector<long> isb_, isl_, bth, index; 
    std::vector<std::vector<double>> pmc; 
    for (size_t x(0); x < met_.size(); ++x){
        long bl = x; 
        for (size_t y(0); y < quarks[x].size(); ++y){
            particle_template* bq = quarks[x][y]; 
            if (!bq -> is_b){continue;}
            index.push_back(long(y)); 
            isl_.push_back(long(0)); 
            isb_.push_back(long(bq -> is_b)); 
            pmc.push_back(pyc::as_pmc(bq)); 
            bth.push_back(bl); 
        } 

        for (size_t y(0); y < leptons[x].size(); ++y){
            particle_template* lp = leptons[x][y]; 
            if (lp -> is_nu || !lp -> is_lep){continue;}
            index.push_back(long(y)); 
            isl_.push_back(long(leptons[x][y] -> is_lep)); 
            isb_.push_back(long(0)); 
            pmc.push_back(pyc::as_pmc(lp)); 
            bth.push_back(bl); 
        } 
    }

    std::vector<std::pair<neutrino*, neutrino*>> out; 
    out = pyc::nusol::combinatorial(&met_, &phi_, &pmc, &bth, &isb_, &isl_, dev, mT, mW, null, perturb, steps); 
    for (size_t x(0); x < out.size(); ++x){
        neutrino* nu1 = std::get<0>(out[x]); 
        nu1 -> bquark = new particle_template(quarks[x][index[nu1 -> b_idx]]); 
        nu1 -> lepton = new particle_template(leptons[x][index[nu1 -> l_idx]]); 

        neutrino* nu2 = std::get<1>(out[x]); 
        nu2 -> bquark = new particle_template(quarks[x][index[nu2 -> b_idx]]); 
        nu2 -> lepton = new particle_template(leptons[x][index[nu2 -> l_idx]]); 
   }
   return out; 
}

