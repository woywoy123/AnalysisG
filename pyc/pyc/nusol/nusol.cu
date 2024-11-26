#include <nusol/nusol.cuh>
#include <nusol/base.cuh>
#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <operators/operators.cuh>
#include <transform/transform.cuh>


std::map<std::string, torch::Tensor> nusol_::BaseDebug(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses){
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
    torch::Tensor isNan   = torch::zeros({dx}, MakeOp(pmc_mu)); 

    torch::Tensor x0p   = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor x0    = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor Sx    = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor Sy    = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor w     = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor om2   = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor eps2  = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor x1    = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor y1    = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor z     = torch::zeros({dx}, MakeOp(pmc_mu)); 


    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "BaseMatrix", [&]{
        _hmatrix_debug<scalar_t><<<blk, thd>>>(
                masses -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     cthe.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                       rt.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),

                pmc_mu -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     m2_l.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     b2_l.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),

                pmc_b -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    m2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    b2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),

                // ---------- outputs ---------- //
                HMatrix.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 H_perp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                  isNan.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
        
                    x0p.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
                     x0.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     Sx.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     Sy.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                      w.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    om2.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                   eps2.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     x1.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     y1.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                      z.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>()); 
    }); 

    std::map<std::string, torch::Tensor> out; 
    out["H"]      = HMatrix; 
    out["H_perp"] = H_perp; 
    out["FailedSols"] = isNan; 
    out["x0p"]  =  x0p;
    out["x0"]   =   x0;
    out["Sx"]   =   Sx;
    out["Sy"]   =   Sy;
    out["w"]    =    w;
    out["om2"]  =  om2;
    out["eps2"] = eps2;
    out["x1"]   =   x1;
    out["y1"]   =   y1;
    out["z"]    =    z;
    out["rt"]   =   rt;
    return out; 
}

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
    out["FailedSols"] = isNan; 
    return out; 
}


std::map<std::string, torch::Tensor> nusol_::Intersection(torch::Tensor* A, torch::Tensor* B, double nulls){
    const unsigned int dx = A -> size({0}); 
    const dim3 thd  = dim3(1, 3, 3);
    const dim3 thdX = dim3(1, 9, 9);
    const dim3 thdY = dim3(1, 27, 3); 

    const dim3 blk  = blk_(dx, 1, 3, 3, 3, 3); 
    const dim3 blkX = blk_(dx, 1, 9, 9, 9, 9); 
    const dim3 blkY = blk_(dx, 1, 27, 27, 3, 3); 

    torch::Tensor inv_A_dot_B = torch::zeros_like(*A); 
    torch::Tensor lines = torch::zeros({dx, 3, 3, 3}, MakeOp(A)); 
    torch::Tensor s_pts = torch::zeros({dx, 9, 3, 3}, MakeOp(A)); 
    torch::Tensor s_dst = torch::zeros({dx, 9, 3, 3}, MakeOp(A)); 

    torch::Tensor a_ = operators_::Determinant(A); 
    torch::Tensor b_ = operators_::Determinant(B);

    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "swp", [&]{
        _swapAB<scalar_t><<<blk, thd>>>(
          inv_A_dot_B.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 A -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 B -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   a_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                   b_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()); 
    });

    //std::tuple<torch::Tensor, torch::Tensor> eig = operators_::Eigenvalue(&inv_A_dot_B); 
    torch::Tensor eig = torch::linalg::eigvals(inv_A_dot_B); 
    torch::Tensor real = torch::real(eig).to(A -> scalar_type()); //std::get<0>(eig); 
    torch::Tensor imag = torch::imag(eig).to(A -> scalar_type()); //std::get<1>(eig); 
    
    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "B-e*A", [&]{
        _factor_degen<scalar_t><<<blkX, thdX>>>(
                 real.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                 imag.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                 A -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 B -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                lines.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                nulls); 
    });

    std::vector<signed long> dim313 = {-1, 3, 1, 3}; 
    std::vector<signed long> dim133 = {-1, 1, 3, 3}; 
    torch::Tensor V = torch::cross(lines.view(dim313), A -> view(dim133), 3); 
    V = torch::transpose(V, 2, 3);
    V = std::get<1>(torch::linalg::eig(V)); 
    V = torch::transpose(V, 2, 3).view({-1, 9, 3, 3}); 
    real = torch::real(V);
    imag = torch::imag(V); 

    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "intersection", [&]{
        _intersections<scalar_t><<<blkY, thdY>>>(
                 real.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                 imag.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                 A -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                lines.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                s_pts.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                s_dst.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                nulls); 
    });

    torch::Tensor sols = torch::zeros({dx, 18, 3}, MakeOp(A)); 
    torch::Tensor solx = torch::zeros({dx, 18, 1}, MakeOp(A)); 
    torch::Tensor idx = std::get<1>(s_pts.sort(-2, false)); 
    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "sorted", [&]{
        _solsx<scalar_t><<<blkY, thdY>>>(
                s_pts.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                s_dst.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                 sols.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 solx.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                  idx.packed_accessor64<long    , 4, torch::RestrictPtrTraits>()); 
    });
 
    std::map<std::string, torch::Tensor> out;
    out["solutions"] = sols; 
    out["distances"] = solx; 
    return out; 
}

std::map<std::string, torch::Tensor> nusol_::Nu(
        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
        torch::Tensor* masses, torch::Tensor* sigma , double null
){
    torch::Tensor H = nusol_::BaseMatrix(pmc_b, pmc_mu, masses)["H"];
    torch::Tensor X = torch::zeros_like(H); 
    torch::Tensor M = torch::zeros_like(H); 
    torch::Tensor Unit = torch::zeros_like(H); 
   
    const unsigned int dx = pmc_b -> size({0}); 
    const dim3 thd = dim3(1, 3, 3);
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "Nu", [&]{
        _nu_init_<scalar_t><<<blk, thd>>>(
                 sigma -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        M.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        Unit.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());

    }); 

    std::map<std::string, torch::Tensor> out = nusol_::Intersection(&M, &Unit, null); 

    const dim3 thN = dim3(1, 18, 3);
    const dim3 blN = blk_(dx, 1, 18, 18, 3, 3); 
    torch::Tensor nu   = torch::zeros({dx, 18, 3}, MakeOp(pmc_b)); 
    torch::Tensor chi2 = torch::zeros({dx, 18, 1}, MakeOp(pmc_b)); 
    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "Nu", [&]{
        _chi2<scalar_t><<<blN, thN>>>(
         out["solutions"].packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
         out["distances"].packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       nu.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     chi2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
    }); 

    out["X"] = X; 
    out["M"] = M; 
    out["nu"] = nu; 
    out["chi2"] = chi2;
    return out; 
}

std::map<std::string, torch::Tensor> nusol_::NuNu(
            torch::Tensor* pmc_b1,  torch::Tensor* pmc_b2, 
            torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy,  torch::Tensor* masses, double null
){
    std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(pmc_b1, pmc_mu1, masses);
    torch::Tensor H1_inv = std::get<0>(operators_::Inverse(&H1_m["H_perp"])); 
    torch::Tensor H1_    = H1_m["H"]; 

    std::map<std::string, torch::Tensor> H2_m = nusol_::BaseMatrix(pmc_b2, pmc_mu2, masses);
    torch::Tensor H2_inv = std::get<0>(operators_::Inverse(&H2_m["H_perp"])); 
    torch::Tensor H2_    = H2_m["H"]; 

    torch::Tensor fall = (H1_m["FailedSols"] + H2_m["FailedSols"]) == false; 
    const unsigned int dx = pmc_b1 -> size({0}); 
    const dim3 thd = dim3(1, 3, 3);
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    
    torch::Tensor S  = torch::zeros({dx, 3, 3}, MakeOp(pmc_b1)); 
    torch::Tensor N  = torch::zeros({dx, 3, 3}, MakeOp(pmc_b1)); 
    torch::Tensor K  = torch::zeros({dx, 3, 3}, MakeOp(pmc_b1)); 
    torch::Tensor n_ = torch::zeros({dx, 3, 3}, MakeOp(pmc_b1)); 
    torch::Tensor K_ = torch::zeros({dx, 3, 3}, MakeOp(pmc_b1)); 

    AT_DISPATCH_ALL_TYPES(pmc_b1 -> scalar_type(), "NuNu", [&]{
        _nunu_init_<scalar_t><<<blk, thd>>>(
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                   H1_inv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H2_inv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                      H1_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                      H2_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),

                       n_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        N.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        K.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       K_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        S.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
    }); 

    std::map<std::string, torch::Tensor> out = nusol_::Intersection(&N, &n_, null); 
    torch::Tensor nu1 = torch::zeros_like(out["solutions"]); 
    torch::Tensor nu2 = torch::zeros_like(out["solutions"]); 
    torch::Tensor v_  = torch::zeros_like(out["solutions"]); 
    torch::Tensor v   = out["solutions"]; 
    torch::Tensor ds  = out["distances"]; 
    torch::Tensor srt = std::get<1>(ds.sort(-2, false));

    const dim3 thN = dim3(1, 18, 3);
    const dim3 blN = blk_(dx, 1, 18, 18, 3, 3); 
    AT_DISPATCH_ALL_TYPES(pmc_b1 -> scalar_type(), "NuNu", [&]{
        _nunu_vp_<scalar_t><<<blN, thN>>>(
                      srt.packed_accessor64<long    , 3, torch::RestrictPtrTraits>(),
                        S.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        K.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       K_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        v.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                       v_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                      nu1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                      nu2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       ds.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); 
    }); 

    out["n_"]  = n_; 
    out["nu1"] = nu1; 
    out["nu2"] = nu2; 
    out["passed"] = fall; 
    return out;  
}


