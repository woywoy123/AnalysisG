#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cuda.h>
#include <vector>

#ifndef PHYSICS_CUDA_KERNEL_H
#define PHYSICS_CUDA_KERNEL_H
#include "kernel.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static const dim3 BLOCKS(const unsigned int threads, const unsigned int len){
    const dim3 blocks( (len + threads -1) / threads ); 
    return blocks; 
}

static const dim3 BLOCKS(
    const unsigned int threads, const unsigned int len, 
    const unsigned int dy,      const unsigned int dz)
{
    const dim3 blocks( (len + threads -1) / threads, dy, dz); 
    return blocks; 
}

static const torch::Tensor format(std::vector<torch::Tensor> i1){
    std::vector<torch::Tensor> ipt; 
    for (unsigned int i(0); i < i1.size(); ++i){ ipt.push_back(i1[i].view({-1, 1})); }
    return torch::cat(ipt, -1).contiguous();  
}


torch::Tensor _Theta(torch::Tensor pmc_){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pmc_.get_device()); 
 
    torch::Tensor pmc = pmc_.contiguous().clone(); 
    CHECK_INPUT(pmc); CHECK_INPUT(pmc_); 
    const unsigned int len = pmc.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len, 3, 1);
    const dim3 blk_ = BLOCKS(threads, len); 
    AT_DISPATCH_FLOATING_TYPES(pmc.scalar_type(), "Theta", ([&]{
        Sq_ij_K<scalar_t><<< blk, threads >>>(
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 0, 2);

        SumK<scalar_t><<< blk_, threads >>>(
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 2); 

        Sqrt_i_K<scalar_t><<< blk_, threads >>>(
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 0); 

        ArcCos_ij_K<scalar_t><<< blk_, threads >>>(
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                pmc_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                len, 0, 2); 
    })); 
    c10::cuda::set_device(current_device);
    return pmc.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
}

torch::Tensor _Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pz.get_device()); 
 
    pz = pz.view({-1, 1}).contiguous(); 
    torch::Tensor pmc = format({px, py, pz}); 

    CHECK_INPUT(pmc); CHECK_INPUT(pz); 
    const unsigned int len = pmc.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len, 3, 1);
    const dim3 blk_ = BLOCKS(threads, len); 
    AT_DISPATCH_FLOATING_TYPES(pmc.scalar_type(), "Theta", ([&]{
        Sq_ij_K<scalar_t><<< blk, threads >>>(
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 0, 2);

        SumK<scalar_t><<< blk_, threads >>>(
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 2); 

        Sqrt_i_K<scalar_t><<< blk_, threads >>>(
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 0); 
        
        ArcCos_ij_K<scalar_t><<< blk_, threads >>>(
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                pz.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                len, 0, 0); 
    })); 
    c10::cuda::set_device(current_device);
    return pmc.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
}

torch::Tensor _DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pmu1.get_device()); 
 
    torch::Tensor eta_phi = torch::cat({
            pmu1.index({torch::indexing::Slice(), torch::indexing::Slice(1, 2)}), 
            pmu2.index({torch::indexing::Slice(), torch::indexing::Slice(1, 2)}), 
            pmu1.index({torch::indexing::Slice(), torch::indexing::Slice(2, 3)}), 
            pmu2.index({torch::indexing::Slice(), torch::indexing::Slice(2, 3)})
    }, -1).contiguous(); 

    CHECK_INPUT(eta_phi); 
    const unsigned int len = eta_phi.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len, 2, 1); 
    const dim3 blk_ = BLOCKS(threads, len); 
    
    AT_DISPATCH_FLOATING_TYPES(eta_phi.scalar_type(), "dR", ([&]{
        delta_K<scalar_t><<< blk, threads >>>(
                eta_phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len); 

        Sum_ij_K<scalar_t><<< blk_, threads >>>(
                eta_phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 0, 2);

        Sqrt_i_K<scalar_t><<< blk_, threads >>>(
                eta_phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 0); 
    }));
    c10::cuda::set_device(current_device);
    return eta_phi.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
}

torch::Tensor _DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(eta1.get_device()); 
 
    torch::Tensor eta_phi = format({eta1, eta2, phi1, phi2}).contiguous(); 
    CHECK_INPUT(eta_phi); 
    const unsigned int len = eta_phi.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len, 2, 1); 
    const dim3 blk_ = BLOCKS(threads, len); 
    
    AT_DISPATCH_FLOATING_TYPES(eta_phi.scalar_type(), "dR", ([&]{
        delta_K<scalar_t><<< blk, threads >>>(
                eta_phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len); 

        Sum_ij_K<scalar_t><<< blk_, threads >>>(
                eta_phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 0, 2); 

        Sqrt_i_K<scalar_t><<< blk_, threads >>>(
                eta_phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), len, 0); 
    }));
    c10::cuda::set_device(current_device);
    return eta_phi.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
}
#endif
