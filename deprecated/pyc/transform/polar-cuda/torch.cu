#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cuda.h>

#ifndef TRANSFORM_CUDA_POLAR_KERNEL_H
#define TRANSFORM_CUDA_POLAR_KERNEL_H
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

static const torch::TensorOptions _MakeOp(torch::Tensor v1)
{
    return torch::TensorOptions().dtype(v1.scalar_type()).device(v1.device()); 
}

torch::Tensor _Pt(torch::Tensor px, torch::Tensor py)
{
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(py.get_device()); 

    px = px.view({-1, 1}).contiguous(); 
    py = py.view({-1, 1}).contiguous(); 
    CHECK_INPUT(px); CHECK_INPUT(py); 

    const torch::TensorOptions op = _MakeOp(py); 
    torch::Tensor pt = torch::zeros_like(px, op);
    const unsigned int len = pt.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len); 

    AT_DISPATCH_FLOATING_TYPES(pt.scalar_type(), "PtK", ([&]
    {
        PtK<scalar_t><<<blk, threads>>>(
            px.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pt.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        ); 
    })); 
    c10::cuda::set_device(current_device);
    return pt; 
}

torch::Tensor _Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
{
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(py.get_device()); 

    px = px.view({-1, 1}).contiguous(); 
    py = py.view({-1, 1}).contiguous(); 
    pz = pz.view({-1, 1}).contiguous();
    CHECK_INPUT(px); CHECK_INPUT(py); CHECK_INPUT(pz);

    const unsigned int len = px.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len); 
    const torch::TensorOptions op = _MakeOp(px); 
    torch::Tensor eta = torch::zeros_like(px, op);

    AT_DISPATCH_FLOATING_TYPES(eta.scalar_type(), "EtaK", ([&]
    {
        EtaK<scalar_t><<<blk, threads>>>(
            px.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pz.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           eta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        ); 
    })); 

    c10::cuda::set_device(current_device);
    return eta; 
}

torch::Tensor _Phi(torch::Tensor px, torch::Tensor py)
{
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(py.get_device()); 

    px = px.view({-1, 1}).contiguous(); 
    py = py.view({-1, 1}).contiguous(); 
    CHECK_INPUT(px); CHECK_INPUT(py);

    const unsigned int len = px.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len); 
    const torch::TensorOptions op = _MakeOp(px); 
    torch::Tensor phi = torch::zeros_like(px, op);

    AT_DISPATCH_FLOATING_TYPES(phi.scalar_type(), "PhiK", ([&]
    {
        PhiK<scalar_t><<<blk, threads>>>(
            px.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        ); 
    })); 

    c10::cuda::set_device(current_device);
    return phi; 
}

torch::Tensor _PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz)
{
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(py.get_device()); 

    px = px.view({-1, 1}).contiguous(); 
    py = py.view({-1, 1}).contiguous();    
    pz = pz.view({-1, 1}).contiguous(); 
    const torch::TensorOptions op = _MakeOp(px); 
    torch::Tensor out = torch::zeros_like(px, op);
    out = torch::cat({out, out, out}, -1).contiguous(); 

    CHECK_INPUT(px); CHECK_INPUT(py); CHECK_INPUT(pz); CHECK_INPUT(out); 

    const unsigned int len = px.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len, 2, 1);  

    AT_DISPATCH_FLOATING_TYPES(px.scalar_type(), "PtEtaPhiK", ([&]
    {
        PtEtaPhiK<scalar_t><<<blk, threads>>>(
            px.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pz.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        );
    })); 

    c10::cuda::set_device(current_device);
    return out; 
}

torch::Tensor _PtEtaPhiE(torch::Tensor pmc)
{
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pmc.get_device()); 

    const torch::TensorOptions op = _MakeOp(pmc); 
    torch::Tensor out = torch::zeros_like(pmc, op); 
    CHECK_INPUT(out); CHECK_INPUT(pmc); 

    const unsigned int len = pmc.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len, 3, 1);  

    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "PtEtaPhiEK", ([&]
    {
        PtEtaPhiEK<scalar_t><<<blk, threads>>>(
            pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        );
    })); 

    c10::cuda::set_device(current_device);
    return out; 
}

#endif
