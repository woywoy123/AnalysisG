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

    v1 = v1.contiguous(); 
    torch::Tensor out = torch::zeros({n_ten, len_i1, len_j2}, _MakeOp(v1)); 
    std::vector<torch::Tensor> tmp; 
    for (unsigned int i(0); i < len_j2; ++i){ tmp.push_back(v1); }
    torch::Tensor tmp_ = torch::cat(tmp, 2); 
    tmp = {}; 
    CHECK_INPUT(out); CHECK_INPUT(v1); 

    const dim3 blk = BLOCKS(threads, len_i1*len_j2, len_co, n_ten); 
    //const dim3 blk_ = BLOCKS(threads, len_i1, len_j2, n_ten); 
    AT_DISPATCH_FLOATING_TYPES(v1.scalar_type(), "MUL", ([&]
    {
        _DotK<scalar_t><<< blk, threads >>>(
                tmp_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                v1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                n_ten, len_i1, len_co, len_j2); 

        //_SumK<scalar_t><<< blk_, threads >>>(
        //         out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
        //        tmp_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
        //        n_ten, len_i1, len_j2, len_j1); 
    })); 

    return tmp_; 
}


#endif
