#include <transform/polar-tensors/polar.h>
#include <operators.h>
#include <physics.h>
#include "nusol.h"

torch::TensorOptions Tooling::MakeOp(torch::Tensor x)
{
    torch::TensorOptions op = torch::TensorOptions().device(x.device()).dtype(x.dtype()); 
    return op; 
}

torch::Tensor Tooling::Pi_2(torch::Tensor x)
{
    torch::TensorOptions op = MakeOp(x); 
    const unsigned int dim_i = x.size(0); 
    torch::Tensor z = torch::zeros({dim_i, 1}, op); 
    return torch::cos(z)/2; 
}

torch::Tensor Tooling::x0(torch::Tensor pmc, torch::Tensor _pm2, torch::Tensor mH2, torch::Tensor mL2)
{
    torch::Tensor e = pmc.index({torch::indexing::Slice(), 3}).view({-1, 1}); 
    return -(mH2 - mL2 - _pm2)/(2*e); 
}

torch::Tensor Tooling::Sigma(torch::Tensor x, torch::Tensor sigma)
{
    sigma = sigma.view({-1, 2, 2}); 
    const unsigned int dim_i = x.size(0);
    const unsigned int dim_i_ = sigma.size(0); 
    const torch::TensorOptions op = Tooling::MakeOp(x); 
    if (dim_i != dim_i_)
    {
        torch::Tensor tmp = torch::ones({dim_i, 2, 2}, op); 
        sigma = tmp * sigma[0]; 
    }
    sigma = torch::inverse(sigma); 
    sigma = torch::pad(sigma, {0, 1, 0, 1}, "constant", 0);
    sigma = torch::transpose(sigma, 1, 2);
    return sigma; 
}

std::map<std::string, torch::Tensor> Tooling::GetMasses(torch::Tensor L, torch::Tensor masses)
{
    const unsigned int dim_i = L.size(0);
    masses = masses.view({-1, 3}); 
    const unsigned int dim_i_ = masses.size(0); 
    if (dim_i != dim_i_)
    {
        torch::Tensor tmp = torch::ones({dim_i, 3}, MakeOp(masses)); 
        masses = tmp*(masses[0]); 
    }
    std::map<std::string, torch::Tensor> out; 
    out["W2"] = torch::pow(masses.index({torch::indexing::Slice(), 0}), 2).view({dim_i, 1});  
    out["T2"] = torch::pow(masses.index({torch::indexing::Slice(), 1}), 2).view({dim_i, 1});  
    out["N2"] = torch::pow(masses.index({torch::indexing::Slice(), 2}), 2).view({dim_i, 1});  
    return out; 
}

torch::Tensor Tooling::Rotation(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor base)
{
    torch::Tensor pmc_b3  = pmc_b.index( {torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 
    torch::Tensor pmc_mu3 = pmc_mu.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 

    torch::Tensor muphi = -Transform::Tensors::Phi(pmc_mu); 
    torch::Tensor theta = Physics::Tensors::Theta(pmc_mu); 
    
    torch::Tensor Ry = Operators::Tensors::Ry(Tooling::Pi_2(theta) - theta); 
    torch::Tensor Rz = Operators::Tensors::Rz(muphi); 
    torch::Tensor Rx = torch::matmul(Rz, pmc_b3.view({-1, 3, 1}));

    Rx = torch::matmul(Ry, Rx.view({-1, 3, 1})); 
    Rx = -torch::atan2(Rx.index({torch::indexing::Slice(), 2}), Rx.index({torch::indexing::Slice(), 1})).view({-1, 1}); 
    Rx = Operators::Tensors::Rx(Rx); 
    
    Rx = torch::transpose(Rx, 1, 2); 
    Ry = torch::transpose(Ry, 1, 2); 
    Rz = torch::transpose(Rz, 1, 2); 
    
    return torch::matmul(torch::matmul(Rz, torch::matmul(Ry, Rx)), base); 
}

torch::Tensor NuSol::Tensor::BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses)
{
    pmc_b  = pmc_b.view( {-1, 4}); 
    pmc_mu = pmc_mu.view({-1, 4}); 
    std::map<std::string, torch::Tensor> mass = Tooling::GetMasses(pmc_b, masses); 
    const torch::TensorOptions op = Tooling::MakeOp(pmc_b); 
    const unsigned int dim_i = pmc_b.size(0); 

    torch::Tensor bB   = Physics::Tensors::Beta(pmc_b); 
    torch::Tensor muB  = Physics::Tensors::Beta(pmc_mu); 
    torch::Tensor muB2 = Physics::Tensors::Beta2(pmc_mu); 
    torch::Tensor muP  = Physics::Tensors::P(pmc_mu); 

    torch::Tensor x0p = Tooling::x0(pmc_b , Physics::Tensors::M2(pmc_b ), mass["T2"], mass["W2"]); 
    torch::Tensor x0  = Tooling::x0(pmc_mu, Physics::Tensors::M2(pmc_mu), mass["W2"], mass["N2"]); 
  
    torch::Tensor pmc_b3  = pmc_b.index( {torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 
    torch::Tensor pmc_mu3 = pmc_mu.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 

    torch::Tensor c_ = Operators::Tensors::CosTheta(pmc_b3, pmc_mu3); 
    torch::Tensor s_ = Operators::Tensors::SinTheta(pmc_b3, pmc_mu3);
    
    torch::Tensor tmp_ = muB / bB; 
    torch::Tensor w_ = ( - tmp_ - c_ ) / s_; 
    torch::Tensor w = ( tmp_ - c_ ) / s_; 
    
    torch::Tensor O2 = w.pow(2) + 1 - muB2;
    torch::Tensor e2 = (mass["W2"] - mass["N2"]) * ( 1 - muB2 ); 
    
    torch::Tensor Sx = (x0 * muB - muP * ( 1 - muB2 )) / muB2; 
    torch::Tensor Sy = ( (x0p / bB) - c_ * Sx ) / s_; 
    
    tmp_ = Sx + w*Sy; 
    torch::Tensor x1 = Sx - tmp_ / O2; 
    torch::Tensor y1 = Sy - tmp_ * (w / O2); 
    torch::Tensor Z = torch::sqrt(torch::relu(x1.pow(2) * O2 - ( Sy - w*Sx ).pow(2) - ( mass["W2"] - x0.pow(2) - e2 ))); 

    torch::Tensor O = torch::sqrt(O2);
    torch::Tensor _0 = torch::zeros({dim_i, 1}, op);
    return torch::cat({ Z / O, _0, x1 - muP, (w*Z)/O, _0, y1, _0, Z, _0}, -1).view({dim_i, 3, 3}); 
}


std::map<std::string, torch::Tensor> NuSol::Tensor::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy, torch::Tensor masses, torch::Tensor sigma)
{
    std::map<std::string, torch::Tensor> output; 
    torch::Tensor base = NuSol::Tensor::BaseMatrix(pmc_b, pmc_mu, masses); 
    torch::Tensor H_base = Tooling::Rotation(pmc_b, pmc_mu, base);
    sigma = Tooling::Sigma(met_xy, sigma); 
    output["M"] = sigma; 
    return output; 
}


std::tuple<torch::Tensor, torch::Tensor> NuSol::Tensor::Intersection(torch::Tensor A, torch::Tensor B, const double null)
{
    return {A, B}; 
}


std::map<std::string, torch::Tensor> NuSol::Tensor::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor MET_xy, torch::Tensor masses, torch::Tensor sigma, const double null)
{
    std::map<std::string, torch::Tensor> output; 

    return output; 
}

std::map<std::string, torch::Tensor> NuSol::Tensor::NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
        torch::Tensor met_x , torch::Tensor met_y, torch::Tensor masses, const double null)
{
    std::map<std::string, torch::Tensor> output; 

    return output; 
}
