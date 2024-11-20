#include <nusol/nusol.cuh>
#include <nusol/base.cuh>

#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <operators/operators.cuh>

torch::Tensor nusol_::BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses){
    torch::Tensor b2_b = physics_::Beta2(pmc_b);
    torch::Tensor b2_l = physics_::Beta2(pmc_mu); 
    torch::Tensor m2_b = physics_::M2(pmc_b);
    torch::Tensor m2_l = physics_::M2(pmc_mu); 
    torch::Tensor cthe = operators_::CosTheta(pmc_b, pmc_mu, 3); 
 
    const unsigned int dx = pmc_b -> size({0}); 
    const dim3 thd = dim3(1, 4, 4);
    const dim3 blk = blk_(dx, 1, 4, 4, 4, 4); 

    unsigned int size = sizeof(nusol)*dx; 
    nusol* solx = nullptr;
    cudaMalloc(&solx, size); 

    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "BaseMatrix", [&]{
        _hmatrix<scalar_t><<<blk, thd, sizeof(nusol)>>>(
                masses -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                cthe.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),

                pmc_mu -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                m2_l.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                b2_l.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),

                pmc_b -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                m2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                b2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                dx, solx); 
    }); 
    cudaFree(solx);

    return *masses; 
}



