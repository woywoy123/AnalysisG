#include <nusol/nusol.cuh>
#include <nusol/base.cuh>

#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <operators/operators.cuh>
#include <transform/transform.cuh>

std::map<std::string, torch::Tensor> nusol_::BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses){
    torch::Tensor b2_b  = physics_::Beta2(pmc_b);
    torch::Tensor b2_l  = physics_::Beta2(pmc_mu); 
    torch::Tensor m2_b  = physics_::M2(pmc_b);
    torch::Tensor m2_l  = physics_::M2(pmc_mu); 
    torch::Tensor theta = physics_::Theta(pmc_mu); 
    torch::Tensor cthe  = operators_::CosTheta(pmc_b, pmc_mu, 3); 

    torch::Tensor px    = pmc_mu -> index({torch::indexing::Slice(), 0});
    torch::Tensor py    = pmc_mu -> index({torch::indexing::Slice(), 1}); 
    torch::Tensor phi   = transform_::Phi(&px, &py);
    torch::Tensor rt    = operators_::RT(pmc_b, &phi, &theta); 
 
    const unsigned int dx = pmc_b -> size({0}); 
    const dim3 thd = dim3(1, 4, 4);
    const dim3 blk = blk_(dx, 1, 4, 4, 4, 4); 

    torch::Tensor HMatrix = torch::zeros({dx, 3, 3}, MakeOp(pmc_mu)); 
    torch::Tensor H_perp  = torch::zeros({dx, 3, 3}, MakeOp(pmc_mu)); 
    torch::Tensor KMatrix = torch::zeros({dx, 4, 4}, MakeOp(pmc_mu)); 
    torch::Tensor A_leps  = torch::zeros({dx, 4, 4}, MakeOp(pmc_mu)); 
    torch::Tensor A_bqrk  = torch::zeros({dx, 4, 4}, MakeOp(pmc_mu)); 
    torch::Tensor isNan   = torch::zeros({dx}, MakeOp(pmc_mu)); 

    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "BaseMatrix", [&]{
        _hmatrix<scalar_t><<<blk, thd>>>(
                masses -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                cthe.packed_accessor64<scalar_t     , 2, torch::RestrictPtrTraits>(),
                rt.packed_accessor64<scalar_t       , 3, torch::RestrictPtrTraits>(),

                pmc_mu -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                m2_l.packed_accessor64<scalar_t     , 2, torch::RestrictPtrTraits>(),
                b2_l.packed_accessor64<scalar_t     , 2, torch::RestrictPtrTraits>(),

                pmc_b -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                m2_b.packed_accessor64<scalar_t    , 2, torch::RestrictPtrTraits>(),
                b2_b.packed_accessor64<scalar_t    , 2, torch::RestrictPtrTraits>(),

                // ---------- outputs ---------- //
                HMatrix.packed_accessor64<scalar_t , 3, torch::RestrictPtrTraits>(),
                H_perp.packed_accessor64<scalar_t  , 3, torch::RestrictPtrTraits>(),
                KMatrix.packed_accessor64<scalar_t , 3, torch::RestrictPtrTraits>(),
                A_leps.packed_accessor64<scalar_t  , 3, torch::RestrictPtrTraits>(),
                A_bqrk.packed_accessor64<scalar_t  , 3, torch::RestrictPtrTraits>(),
                isNan.packed_accessor64<scalar_t   , 1, torch::RestrictPtrTraits>()); 
    }); 
    std::map<std::string, torch::Tensor> out; 
    out["H"] = HMatrix; 
    out["H_perp"] = H_perp; 
    out["K"] = KMatrix;
    out["A_lepton"] = A_leps;
    out["A_bquark"] = A_bqrk; 
    out["FailedSols"] = isNan; 
    return out; 
}


std::map<std::string, torch::Tensor> nusol_::Intersection(torch::Tensor* A, torch::Tensor* B){
    torch::Tensor a_ = operators_::Determinant(A); 
    torch::Tensor b_ = operators_::Determinant(B);

    return {}; 
}



std::map<std::string, torch::Tensor> nusol_::Nu(
        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
        torch::Tensor* masses, torch::Tensor* sigma , double null
){
    torch::Tensor H = nusol_::BaseMatrix(pmc_b, pmc_mu, masses)["H"];
    torch::Tensor X = torch::zeros_like(H); 
    torch::Tensor M = torch::zeros_like(H); 
   
    const unsigned int dx = pmc_b -> size({0}); 
    const dim3 thd = dim3(1, 3, 3);
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "Nu", [&]{
        _nu_init_<scalar_t><<<blk, thd>>>(
                 sigma -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        M.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());

    }); 

    std::map<std::string, torch::Tensor> out;
    out = nusol_::Intersection(&M, &X); 
    out["X"] = X; 
    out["M"] = M; 
    return out; 
}


