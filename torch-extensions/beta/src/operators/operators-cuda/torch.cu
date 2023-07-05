#ifndef OPERATORS_CUDA_KERNEL_H
#define OPERATORS_CUDA_KERNEL_H
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

torch::TensorOptions _MakeOp(torch::Tensor v1)
{
	return torch::TensorOptions().dtype(v1.scalar_type()).device(v1.device()); 
}

torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2)
{
    torch::Tensor out = v1.contiguous().clone();
    CHECK_INPUT(out); CHECK_INPUT(v2);  
    const unsigned int len_i = v1.size(0); 
    const unsigned int len_j = v1.size(1); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len_i, len_j); 
    const dim3 blk_ = BLOCKS(threads, len_i); 
    AT_DISPATCH_FLOATING_TYPES(v1.scalar_type(), "DOT", ([&]
    {
        _DotK<scalar_t><<< blk, threads >>>( 
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                len_i, len_j); 
        _SumK<scalar_t><<< blk_, threads >>>(
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                len_i, len_j); 
    })); 
    return out.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
}

torch::Tensor _Mul(torch::Tensor v1, torch::Tensor v2)
{
    const unsigned int  n_ten = v1.size(0); 
    const unsigned int len_i1 = v1.size(1); 
    const unsigned int len_co = v1.size(2); 
    const unsigned int len_j2 = v2.size(2); 
    const unsigned int threads = 1024; 
    const unsigned int grid = len_i1*len_j2; 

    v2 = v2.contiguous(); 
    v1 = v1.contiguous(); 
    torch::Tensor out =  torch::zeros({n_ten, len_i1, len_j2}, _MakeOp(v1)); 
    torch::Tensor tmp_ = torch::zeros({n_ten, len_i1, len_j2*len_co}, _MakeOp(v1)); 
    CHECK_INPUT(v2); CHECK_INPUT(v1); 

    const dim3 blk = BLOCKS(threads, grid, len_co, n_ten); 
    const dim3 blk_ = BLOCKS(threads, len_i1, len_j2, n_ten); 
    AT_DISPATCH_FLOATING_TYPES(v2.scalar_type(), "MUL", ([&]
    {
        _DotK<scalar_t><<< blk, threads >>>(
                tmp_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                v1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                v2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                n_ten, len_i1, len_co, grid); 

        _SumK<scalar_t><<< blk_, threads >>>(
                 out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                tmp_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                n_ten, len_i1, len_j2, len_co); 
    })); 

    return out; 
}

torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2)
{
    const unsigned int x = v1.size(0); 
    const unsigned int y = v1.size(1); 
    const unsigned int threads = 1024; 

    v2 = v2.contiguous(); 
    v1 = v1.contiguous(); 
    torch::Tensor tmp = torch::zeros({x, y, 3}, _MakeOp(v1));
    torch::Tensor out = torch::zeros({x, 3}, _MakeOp(v1));  
    CHECK_INPUT(v1); CHECK_INPUT(v2); 
    const dim3 blk = BLOCKS(threads, x, y, 3); 
    const dim3 blk_ = BLOCKS(threads, x, 3); 
    const dim3 blk__ = BLOCKS(threads, x); 
    AT_DISPATCH_FLOATING_TYPES(v1.scalar_type(), "COSTHETA", ([&]
    {
        _DotK<scalar_t><<< blk, threads >>>(
                tmp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                x, y);
        _SumK<scalar_t><<< blk_, threads >>>(
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                tmp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                x, y); 
        _CosThetaK<scalar_t><<< blk__, threads >>>(
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                x); 
    })); 
    return out.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
}

torch::Tensor _SinTheta(torch::Tensor v1, torch::Tensor v2)
{
    const unsigned int x = v1.size(0); 
    const unsigned int y = v1.size(1); 
    const unsigned int threads = 1024; 

    v2 = v2.contiguous(); 
    v1 = v1.contiguous(); 
    torch::Tensor tmp = torch::zeros({x, y, 3}, _MakeOp(v1));
    torch::Tensor out = torch::zeros({x, 3}, _MakeOp(v1));  
    CHECK_INPUT(v1); CHECK_INPUT(v2); 
    const dim3 blk = BLOCKS(threads, x, y, 3); 
    const dim3 blk_ = BLOCKS(threads, x, 3); 
    const dim3 blk__ = BLOCKS(threads, x); 
    AT_DISPATCH_FLOATING_TYPES(v1.scalar_type(), "SINTHETA", ([&]
    {
        _DotK<scalar_t><<< blk, threads >>>(
                tmp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                v1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                v2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                x, y);
        _SumK<scalar_t><<< blk_, threads >>>(
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                tmp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                x, y); 
        _SinThetaK<scalar_t><<< blk__, threads >>>(
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                x); 
    })); 
    return out.index({torch::indexing::Slice(), 0}).view({-1, 1}); 
}

torch::Tensor _Rot(torch::Tensor angle, const unsigned int dim)
{
    const unsigned int x = angle.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, x, 3, 3); 
    angle = angle.view({-1, 1}).contiguous(); 
    CHECK_INPUT(angle); 

    torch::Tensor out = torch::zeros({x, 3, 3}, _MakeOp(angle));
    AT_DISPATCH_FLOATING_TYPES(angle.scalar_type(), "Rot", ([&]
    {
        _RotK<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                angle.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                x, dim); 
    })); 
    return out; 
}

torch::Tensor _CoFactors(torch::Tensor matrix)
{
    const unsigned int x = matrix.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, x, 3, 3);  
    torch::Tensor out = torch::zeros_like(matrix);
    CHECK_INPUT(matrix);  

    AT_DISPATCH_FLOATING_TYPES(matrix.scalar_type(), "Cofactor", ([&]
    {
        _CoFactorK<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                matrix.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                x, 3); 
    })); 

    return out; 
}

torch::Tensor _Det(torch::Tensor matrix)
{
    const unsigned int x = matrix.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, x, 1, 3);
    const dim3 blk_ = BLOCKS(threads, x);
    torch::Tensor tmp = torch::zeros({x, 1, 3}, _MakeOp(matrix));
    torch::Tensor out = torch::zeros({x, 1}, _MakeOp(matrix));
    AT_DISPATCH_FLOATING_TYPES(matrix.scalar_type(), "Determinant", ([&]
    {
        _CoFactorK<scalar_t><<< blk, threads >>>(
                tmp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                matrix.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                x, 1); 

        _DetDotK<scalar_t><<< blk, threads >>>(
                tmp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                tmp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                matrix.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                x);

        _DetSumK<scalar_t><<< blk_, threads >>>(
                out.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                tmp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                x);
    })); 
    return out; 
}

torch::Tensor _Inv(torch::Tensor matrix)
{
    const unsigned int x = matrix.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk_ = BLOCKS(threads, x);
    const dim3 blk__ = BLOCKS(threads, x, 1, 3);
    const dim3 blk = BLOCKS(threads, x, 3, 3);
    torch::Tensor coef = torch::zeros({x, 3, 3}, _MakeOp(matrix));
    torch::Tensor det = torch::zeros({x, 1}, _MakeOp(matrix));
    torch::Tensor out = torch::zeros({x, 3, 3}, _MakeOp(matrix));
    AT_DISPATCH_FLOATING_TYPES(matrix.scalar_type(), "Determinant", ([&]
    {
        _CoFactorK<scalar_t><<< blk, threads >>>(
                  coef.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                matrix.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                x, 3); 

        _DetDotK<scalar_t><<< blk__, threads >>>(
                   out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                  coef.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                matrix.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                x);

        _DetSumK<scalar_t><<< blk_, threads >>>(
                det.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                x);
    
        _InvK<scalar_t><<< blk, threads >>>(
                 out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                 det.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                coef.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                x);
    })); 
    return out; 
}









#endif
