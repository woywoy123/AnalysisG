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

torch::Tensor _P2(torch::Tensor Pmc)
{
    Pmc = Pmc.contiguous().clone(); 
    CHECK_INPUT(Pmc); 
    const unsigned int len = Pmc.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk_ = BLOCKS(threads, len, 3, 1); 
    const dim3 blk  = BLOCKS(threads, len); 
    AT_DISPATCH_FLOATING_TYPES(Pmc.scalar_type(), "Px2Py2Pz2K", ([&]
    { 
        Px2Py2Pz2K<scalar_t><<< blk_, threads >>>(
            Pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len, 3); 
        SumK<scalar_t><<< blk, threads >>>(
            Pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            len, 3); 
    })); 
    return Pmc.index({torch::indexing::Slice(), 0}).view({-1, 1}).clone(); 
}

torch::Tensor _P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
{
    px =  px.view({-1, 1}).contiguous(); 
    py =  px.view({-1, 1}).contiguous(); 
    pz =  px.view({-1, 1}).contiguous(); 
    torch::Tensor Pmc = torch::cat({px, py, pz}, -1); 
    return _P2(Pmc); 
}

torch::Tensor _P(torch::Tensor Pmc)
{
    Pmc = Pmc.contiguous().clone(); 
    CHECK_INPUT(Pmc); 
    const unsigned int len = Pmc.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk_ = BLOCKS(threads, len, 3, 1); 
    const dim3 blk  = BLOCKS(threads, len); 
    const dim3 blk_s = BLOCKS(threads, len, 1, 1); 
    AT_DISPATCH_FLOATING_TYPES(Pmc.scalar_type(), "Px2Py2Pz2K", ([&]
    { 
        Px2Py2Pz2K<scalar_t><<< blk_, threads >>>(
            Pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len, 3); 
        SumK<scalar_t><<< blk, threads >>>(
            Pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            len, 3); 
        SqrtK<scalar_t><<< blk_s, threads >>>(
            Pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            len, 0, 0);  
    })); 
    return Pmc.index({torch::indexing::Slice(), 0}).view({-1, 1}).clone(); 
}

torch::Tensor _P(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
{
    px =  px.view({-1, 1}).contiguous(); 
    py =  px.view({-1, 1}).contiguous(); 
    pz =  px.view({-1, 1}).contiguous(); 
    torch::Tensor Pmc = torch::cat({px, py, pz}, -1); 
    return _P(Pmc); 
}



#endif
