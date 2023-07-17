#include <transform/polar-cuda/polar.h>
#include <torch/torch.h>
#include <operators.h>
#include <physics.h>
#include <vector>

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

torch::Tensor _Shape_Matrix(torch::Tensor inpt, std::vector<long> vec)
{
    const torch::TensorOptions op = _MakeOp(inpt); 
    const unsigned int len_i = inpt.size(0); 
    const unsigned int len_j = vec.size(); 
    torch::Tensor out = torch::zeros_like(inpt); 
    torch::Tensor vecT = torch::zeros({1, 1, len_j}, op).to(torch::kCPU); 
    for (unsigned int i(0); i < len_j; ++i){ vecT[0][0][i] += vec[i]; }
    vecT = vecT.to(inpt.device()); 

    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len_i, len_j, len_j);
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]
    {
        _ShapeKernel<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                vecT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, len_j, len_j, false); 
    })); 
    return out; 
} 

torch::Tensor _Expand_Matrix(torch::Tensor inpt, torch::Tensor source)
{
    const unsigned int threads = 1024; 
    const unsigned int len_i = inpt.size(0);
    const unsigned int len_k = source.size(1); 
    source = source.view({source.size(0), len_k, -1}); 
    const unsigned int len_j = source.size(2); 
    const dim3 blk = BLOCKS(threads, len_i, len_k, len_j);
    torch::Tensor out = torch::zeros_like(inpt); 

    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]
    {
        _ShapeKernel<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                source.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, (len_k > inpt.size(1)) ? inpt.size(1) : len_k, len_j, true); 
    })); 
    return out; 
} 

torch::Tensor _Base_Matrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu)
{
    const unsigned int threads = 1024; 
    const unsigned int len_i = pmc_b.size(0); 
    const unsigned int len_m = masses_W_top_nu.size(0); 

    torch::Tensor beta2_b   = Physics::CUDA::Beta2(pmc_b); 
    torch::Tensor mass2_b   = Physics::CUDA::M2(pmc_b); 
    
    torch::Tensor beta2_mu  = Physics::CUDA::Beta2(pmc_mu);
    torch::Tensor mass2_mu  = Physics::CUDA::M2(pmc_mu); 
    
    torch::Tensor costheta  = Operators::CUDA::CosTheta(pmc_b, pmc_mu, 3);

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
               len_i, len_m); 
    })); 
    return out; 
} 

torch::Tensor _Base_Matrix_H(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses)
{
    torch::Tensor base = _Base_Matrix(pmc_b, pmc_mu, masses); 
    torch::Tensor phi   = Transform::CUDA::Phi(pmc_mu); 
    torch::Tensor theta = Physics::CUDA::Theta(pmc_mu); 

    torch::Tensor Rx = _Expand_Matrix(base, pmc_b); 
    torch::Tensor Rz = torch::zeros_like(base); 
    torch::Tensor Ry = torch::zeros_like(base); 

    torch::Tensor RzT = torch::zeros_like(base); 
    torch::Tensor RyT = torch::zeros_like(base); 
    torch::Tensor RxT = torch::zeros_like(base); 

    const unsigned int dim_i = Rx.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, dim_i, 6, 3); 
    const dim3 blk_ = BLOCKS(threads, dim_i, 3, 3); 

    AT_DISPATCH_FLOATING_TYPES(phi.scalar_type(), "BaseMatrixH", ([&]
    {
        _Base_Matrix_H_Kernel<scalar_t><<< blk, threads >>>(
                Ry.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                Rz.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

                RyT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                RzT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

                phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
              theta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                dim_i); 

        Rx = Operators::CUDA::Mul(Rz, Rx); 
        Rx = Operators::CUDA::Mul(Ry, Rx);

        _Base_Matrix_H_Kernel<scalar_t><<< blk_, threads >>>(
                 Rx.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                RxT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                dim_i); 
        
        Rx = Operators::CUDA::Mul(RyT, RxT); 
        Rx = Operators::CUDA::Mul(RzT, Rx); 
    })); 
    return Operators::CUDA::Mul(Rx, base);
}

torch::Tensor _DotMatrix(torch::Tensor MET_xy, torch::Tensor H, torch::Tensor Shape)
{
    const unsigned int dim_i = MET_xy.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3);

    MET_xy = _Expand_Matrix(H, MET_xy); 
    CHECK_INPUT(MET_xy); 
 
    torch::Tensor X = torch::zeros_like(H);
    torch::Tensor dNu = torch::zeros_like(H); 
    AT_DISPATCH_FLOATING_TYPES(MET_xy.scalar_type(), "NuMatrix", ([&]
    {
        _V0_deltaK<scalar_t><<< blk, threads >>>(
                 X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
               dNu.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            MET_xy.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
             Shape.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                 H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 dim_i); 
    })); 

    return Operators::CUDA::Mul(X, dNu); 
}

torch::Tensor _Intersection(torch::Tensor A, torch::Tensor B)
{
    torch::Tensor det_A = Operators::CUDA::Determinant(A); 
    torch::Tensor det_B = Operators::CUDA::Determinant(B); 
    const unsigned int dim_i = det_A.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3); 
    AT_DISPATCH_FLOATING_TYPES(det_A.scalar_type(), "Swap", ([&]
    {
        _SwapAB_K<scalar_t><<< blk, threads >>>(
                det_A.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                det_B.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                A.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                B.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                dim_i); 
    })); 
    torch::Tensor x; 
    x = Operators::CUDA::Inverse(A);
    x = Operators::CUDA::Dot(x, B);  
    return x; 
}

#endif