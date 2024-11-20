#include <cutils/utils.cuh>
#include <nusol/base.cuh>
#include <shape.cuh>

torch::Tensor ShapeMatrix(torch::Tensor* inpt, std::vector<long> vec){
    const unsigned int len_i = inpt -> size(0); 
    const unsigned int len_j = vec.size(); 
    const unsigned int threads = 1024; 
    torch::TensorOptions op = MakeOp(inpt); 

    torch::Tensor out  = torch::zeros_like(*inpt); 
    torch::Tensor vecT = torch::zeros({1, 1, len_j}, op);
    for (size_t i(0); i < len_j; ++i){ vecT[0][0][i] += vec[i]; }

    const dim3 blk = BLOCKS(threads, len_i, len_j, len_j);
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]{
        _shape_kernel<scalar_t><<< blk, threads >>>(
                 out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                vecT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, len_j, len_j, false); 
    })); 
    return out; 
} 

torch::Tensor ExpandMatrix(torch::Tensor* inpt, torch::Tensor* source){
    const unsigned int threads = 1024; 
    const unsigned int len_i = inpt -> size(0);
    const unsigned int len_k = source -> size(1); 
    torch::Tensor src = source -> view({source -> size(0), len_k, -1}).clone(); 
    torch::Tensor out = torch::zeros_like(*inpt); 

    const unsigned int len_j = src.size(2); 
    const unsigned int dz = (len_k > inpt -> size(1)) ? inpt -> size(1) : len_k; 

    const dim3 blk = BLOCKS(threads, len_i, len_k, len_j);
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ExpandMatrix", ([&]{
        _shape_kernel<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                src.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, dz, len_j, true); 
    })); 
    return out; 
}

torch::Tensor BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* mWtnu){



    return *mWtnu; 
}
