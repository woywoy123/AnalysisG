#ifndef PHYSICS_CUDA_KERNEL_H
#define PHYSICS_CUDA_KERNEL_H
#include <torch/torch.h>
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
    const unsigned int threads, const unsigned int len, 
    const unsigned int dy,      const unsigned int dz 
)
{
    const dim3 blocks( (len + threads -1) / threads, dy, dz); 
    return blocks; 
}


torch::Tensor _P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
{
    px =  px.view({-1, 1}).contiguous(); 
    py =  px.view({-1, 1}).contiguous(); 
    pz =  px.view({-1, 1}).contiguous(); 
    torch::Tensor pmc = torch::cat({px, py, pz}, -1); 
    torch::Tensor p2 = torch::zeros_like(px); 
    CHECK_INPUT(px); CHECK_INPUT(py); CHECK_INPUT(pz);
    CHECK_INPUT(p2); CHECK_INPUT(pmc); 
    
    const unsigned int len = px.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk_ = BLOCKS(threads, len, 3, 1); 
    AT_DISPATCH_FLOATING_TYPES(px.scalar_type(), "Px2Py2Pz2K", ([&]
    { 
        Px2Py2Pz2K<scalar_t><<< blk_, threads >>>(
            pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len); 
    })); 

    return pmc; 
}

torch::Tensor _P2(torch::Tensor Pmc)
{
    Pmc = Pmc.contiguous(); 
    torch::Tensor p2 = torch::zeros_like(Pmc); 
    CHECK_INPUT(p2); CHECK_INPUT(Pmc); 
    
    const unsigned int len = Pmc.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk_ = BLOCKS(threads, len, 3, 1); 
    AT_DISPATCH_FLOATING_TYPES(Pmc.scalar_type(), "Px2Py2Pz2K", ([&]
    { 
        Px2Py2Pz2K<scalar_t><<< blk_, threads >>>(
            Pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len); 
    })); 

    return Pmc; 
}

#endif
