#include <torch/torch.h>
#include <physics.h>
#include <operators.h>

#ifndef NUSOL_CUDA_KERNEL_H
#define NUSOL_CUDA_KERNEL_H
#include "kernel.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static const dim3 BLOCKS(const unsigned int threads, const unsigned int len)
{
    const dim3 blocks( (len + threads -1) / threads ); 
    return blocks; 
}

static const dim3 BLOCKS(
    const unsigned int threads, const unsigned int len, const unsigned int dy)
{
    const dim3 blocks( (len + threads -1) / threads, dy); 
    return blocks; 
}

static const dim3 BLOCKS(
    const unsigned int threads, const unsigned int len, 
    const unsigned int dy, const unsigned int dz)
{
    const dim3 blocks( (len + threads -1) / threads, dy, dz); 
    return blocks; 
}

static const torch::TensorOptions _MakeOp(torch::Tensor v1)
{
	return torch::TensorOptions().dtype(v1.scalar_type()).device(v1.device()); 
}

torch::Tensor _Base_Matrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu)
{
    const unsigned int threads = 1024; 
    const unsigned int len_i   = pmc_b.size(0); 

    torch::Tensor beta2_b   = _Beta2(pmc_b); 
    torch::Tensor mass2_b   = _M2(pmc_b); 
    
    torch::Tensor beta2_mu  = _Beta2(pmc_mu);
    torch::Tensor mass2_mu  = _M2(pmc_mu); 
    
    torch::Tensor costheta  = _CosTheta(pmc_b, pmc_mu, 3);

    // [Z/Om, 0, x1 - p_mu], [ w * Z/Om, 0, y1 ], [0, Z, 0]
    torch::Tensor out = torch::zeros({len_i, 3, 3}, _MakeOp(costheta)); 
    const dim3 blk = BLOCKS(threads, len_i, 3, 3); 
    AT_DISPATCH_FLOATING_TYPES(costheta.scalar_type(), "BaseMatrix", ([&]
    {
        _H_Base<scalar_t><<< blk, threads>>>(
                    out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

                beta2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                mass2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                  pmc_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 

               beta2_mu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               mass2_mu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                 pmc_mu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 

               costheta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
        masses_W_top_nu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 

               len_i); 
    })); 
    return out; 
} 

torch::Tensor _Nu_Matrix(torch::Tensor MET_xy, torch::Tensor Sigma, torch::Tensor H)
{
    CHECK_INPUT(MET_xy); 
    Sigma = Sigma.view({-1, 2, 2}); 
    const unsigned int dim_i = MET_xy.size(0); 
    const unsigned int sig_i = Sigma.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3); 
    const torch::TensorOptions op = _MakeOp(Sigma); 
    torch::Tensor z = torch::zeros({ sig_i, 2, 1 }, op);
    Sigma = torch::cat({Sigma, z}, -1);
    Sigma = torch::cat({Sigma, torch::cat({z.view({-1, 1, 2}), torch::ones({ sig_i, 1, 1 }, op)}, -1)}, 1);
    Sigma = Sigma.view({-1, 3, 3});  
    Sigma = _Inv(Sigma); 
    torch::Tensor X = torch::zeros_like(H);
    AT_DISPATCH_FLOATING_TYPES(MET_xy.scalar_type(), "NuMatrix", ([&]
    {
        _Nu_deltaK<scalar_t><<< blk, threads >>>(
                 X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            MET_xy.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
             Sigma.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                 H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 dim_i, sig_i); 
    })); 

    return X; 
}





#endif
