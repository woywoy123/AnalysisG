#ifndef H_TRANSFORM_CUDA_POLAR
#define H_TRANSFORM_CUDA_POLAR

#include <torch/torch.h>
torch::Tensor _Pt(torch::Tensor px, torch::Tensor py);
torch::Tensor _Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
torch::Tensor _Phi(torch::Tensor px, torch::Tensor py);
torch::Tensor _PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
torch::Tensor _PtEtaPhiE(torch::Tensor Pmc);

namespace transform {
    namespace cuda {
        torch::Tensor Pclip(torch::Tensor inpt, int dim);
        torch::Tensor Pt(torch::Tensor px, torch::Tensor py);
        torch::Tensor Pt(torch::Tensor pmc);
        torch::Tensor Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
        torch::Tensor Eta(torch::Tensor pmc);
        torch::Tensor Phi(torch::Tensor px, torch::Tensor py);
        torch::Tensor Phi(torch::Tensor pmc);
        torch::Tensor PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
        torch::Tensor PtEtaPhi(torch::Tensor pmc);
        torch::Tensor PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
        torch::Tensor PtEtaPhiE(torch::Tensor pmc);
    }
}

#endif
