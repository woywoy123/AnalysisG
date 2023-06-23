#ifndef H_TRANSFORM_CUDA_POLAR
#define H_TRANSFORM_CUDA_POLAR

#include <torch/torch.h>
torch::Tensor _Pt(torch::Tensor px, torch::Tensor py); 
torch::Tensor _Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor _Phi(torch::Tensor px, torch::Tensor py); 
torch::Tensor _PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
torch::Tensor _PtEtaPhiE(torch::Tensor Pmc);

namespace Transform
{
    namespace CUDA
    {
        const torch::Tensor Pt(torch::Tensor px, torch::Tensor py){return _Pt(px, py);} 
        const torch::Tensor Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){return _Eta(px, py, pz);} 
        const torch::Tensor Phi(torch::Tensor px, torch::Tensor py){return _Phi(px, py);} 
        const torch::Tensor PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){return _PtEtaPhi(px, py, pz);}
        const torch::Tensor PtEtaPhiE(torch::Tensor Pmc){return _PtEtaPhiE(Pmc);}
    }
}

#endif
