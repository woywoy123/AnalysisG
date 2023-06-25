#ifndef H_TRANSFORM_TENSORS_POLAR
#define H_TRANSFORM_TENSORS_POLAR
#include <torch/torch.h>

namespace Transform
{
    namespace Tensors
    {
        torch::Tensor Pt(torch::Tensor px, torch::Tensor py); 
	    torch::Tensor Phi(torch::Tensor px, torch::Tensor py);
	    torch::Tensor Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz); 
	    torch::Tensor PtEta(torch::Tensor pt, torch::Tensor pz); 
	    torch::Tensor PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz);
	    torch::Tensor PtEtaPhiE(torch::Tensor Pmc);
    }
}

#endif
