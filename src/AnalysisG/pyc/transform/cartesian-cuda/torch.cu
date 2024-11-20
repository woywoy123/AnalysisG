#include <c10/cuda/CUDAFunctions.h>
#include <cutils/utils.cuh>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cuda.h>

#ifndef TRANSFORM_CARTESIAN_CUDA_H
#define TRANSFORM_CARTESIAN_CUDA_H
#include <transform/cartesian.cuh>



#include "kernel.cu"

torch::Tensor _Px(torch::Tensor pt, torch::Tensor phi){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pt.get_device()); 

    pt = pt.view({-1, 1}).contiguous(); 
    phi = phi.view({-1, 1}).contiguous();
    CHECK_INPUT(pt); CHECK_INPUT(phi);  
    const torch::TensorOptions op = MakeOp(&pt); 

    torch::Tensor px = torch::zeros_like(pt, op);
    const unsigned int threads = 1024;   
    const unsigned int len = pt.size(0); 
    const dim3 blk = BLOCKS(threads, len); 
    
    AT_DISPATCH_FLOATING_TYPES(pt.scalar_type(), "PxK", ([&]
    {
        PxK<scalar_t><<<blk, threads>>>(
            pt.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            px.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        );
    })); 

    c10::cuda::set_device(current_device);
    return px;  
}

torch::Tensor _Py(torch::Tensor pt, torch::Tensor phi){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pt.get_device()); 

    pt = pt.view({-1, 1}).contiguous(); 
    phi = phi.view({-1, 1}).contiguous(); 
    CHECK_INPUT(pt); CHECK_INPUT(phi);  
    const torch::TensorOptions op = MakeOp(&pt); 

    torch::Tensor py = torch::zeros_like(pt, op);
    const unsigned int threads = 1024;   
    const unsigned int len = pt.size(0); 
    const dim3 blk = BLOCKS(threads, len); 
    
    AT_DISPATCH_FLOATING_TYPES(pt.scalar_type(), "PyK", ([&]
    {
        PyK<scalar_t><<<blk, threads>>>(
            pt.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        );
    })); 

    c10::cuda::set_device(current_device);
    return py;  
}

torch::Tensor _Pz(torch::Tensor pt, torch::Tensor eta){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pt.get_device()); 

    pt = pt.view({-1, 1}).contiguous(); 
    eta = eta.view({-1, 1}).contiguous(); 
    CHECK_INPUT(pt); CHECK_INPUT(eta);  
    const torch::TensorOptions op = MakeOp(&pt); 

    torch::Tensor pz = torch::zeros_like(pt, op);
    const unsigned int threads = 1024;   
    const unsigned int len = pt.size(0); 
    const dim3 blk = BLOCKS(threads, len); 
    
    AT_DISPATCH_FLOATING_TYPES(pt.scalar_type(), "PzK", ([&]
    {
        PzK<scalar_t><<<blk, threads>>>(
            pt.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           eta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pz.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        );
    })); 

    c10::cuda::set_device(current_device);
    return pz;  
}

torch::Tensor _PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pt.get_device()); 

    pt  = pt.view({-1, 1}).contiguous(); 
    eta = eta.view({-1, 1}).contiguous(); 
    phi = phi.view({-1, 1}).contiguous();     
    const torch::TensorOptions op = MakeOp(&pt); 

    torch::Tensor out = torch::zeros_like(pt, op);
    out = torch::cat({out, out, out}, -1).contiguous(); 
    CHECK_INPUT(pt); CHECK_INPUT(eta); CHECK_INPUT(phi); CHECK_INPUT(out); 

    const unsigned int len = pt.size(0); 
    const unsigned int threads = 1024;   
    const dim3 blk = BLOCKS(threads, len, 3, 1); 
    
    AT_DISPATCH_FLOATING_TYPES(pt.scalar_type(), "PxPyPzK", ([&]
    {
        PxPyPzK<scalar_t><<<blk, threads>>>(
            pt.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           eta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        );
    })); 

    c10::cuda::set_device(current_device);
    return out;  
}

torch::Tensor _PxPyPzE(torch::Tensor pmu){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(pmu.get_device()); 

    const torch::TensorOptions op = MakeOp(&pmu); 
    torch::Tensor out = torch::zeros_like(pmu, op);
    CHECK_INPUT(out); CHECK_INPUT(pmu); 

    const unsigned int len = out.size(0); 
    const unsigned int threads = 1024;   
    const dim3 blk = BLOCKS(threads, len, 4, 1); 
    
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "PxPyPz3K", ([&]
    {
        PxPyPzEK<scalar_t><<<blk, threads>>>(
            pmu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            len
        );
    })); 
    c10::cuda::set_device(current_device);
    return out;  
}

#endif
