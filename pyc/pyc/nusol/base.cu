#include <nusol/base.cuh>
#include <cutils/utils.cuh>
#include <atomic/cuatomic.cuh>

torch::Tensor nusol_::ShapeMatrix(torch::Tensor* inpt, std::vector<long> vec){
    const unsigned int dx = inpt -> size({0}); 
    const unsigned int dy = inpt -> size({1}); 
    const unsigned int dz = vec.size(); 
    torch::Tensor out = torch::zeros_like(*inpt); 

    long* diag = nullptr;
    cudaMalloc(&diag, sizeof(long)*dz); 
    cudaMemcpy(diag, vec.data(), sizeof(long)*dz, cudaMemcpyHostToDevice); 

    dim3 thd = dim3(_threads, 1, 1); 
    dim3 blk = blk_(dx, _threads, dy, 1, dy, 1); 
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]{
        _shape_matrix<scalar_t><<< blk, thd >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                dx, dy, vec.size(), diag); 
    })); 
    cudaFree(diag); 
    return out; 
}
