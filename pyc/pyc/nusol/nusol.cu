#include <nusol/nusol.cuh>
#include <nusol/base.cuh>
#include <nusol/utils.cuh>
#include <cutils/utils.cuh>
#include <operators/operators.cuh>

std::map<std::string, torch::Tensor> nusol_::combinatorial(
    torch::Tensor* edge_index, torch::Tensor* batch, torch::Tensor* pmc, 
    torch::Tensor* pid, torch::Tensor* met_xy, 
    double  mT, double   mW, double t_pm, double w_pm, 
    long steps, double null, bool gev
){
    const unsigned int btch_dx = batch -> size({0}); 
    const unsigned int evnt_dx = met_xy -> size({0}); 

    const dim3 thdx = dim3(btch_dx);
    const dim3 bldx = blk_(btch_dx, btch_dx); 

    // gather how many particles are there per event and how many b/leps are in the event.
    unsigned int size_b = sizeof(long)*btch_dx*3; 
    torch::Tensor num_events = torch::zeros({evnt_dx}, MakeOp(batch));
    torch::Tensor num_pid    = torch::zeros({evnt_dx, 2}, MakeOp(batch)); 
    AT_DISPATCH_ALL_TYPES(batch -> scalar_type(), "counts", [&]{
        _count<scalar_t><<<bldx, thdx, size_b>>>(
                  batch -> packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    pid -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                num_events.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                   num_pid.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                btch_dx); 
    }); 

    // derive all possible b/l pairs
    long mx = std::get<0>(torch::max(num_events*num_events, {-1})).item<long>(); 
    torch::Tensor num_edges = (num_events*num_events).cumsum(-1) - mx; 

    const unsigned int edge_dx  = edge_index -> size({-1}); 
    const unsigned int edge_idx = edge_dx * mx; 

    const dim3 thcmb  = dim3(mx); 
    const dim3 blkcmb = blk_(edge_idx, mx);
    torch::Tensor   msk = torch::zeros({edge_idx}, MakeOp(batch)); 
    torch::Tensor combi = torch::zeros({edge_idx, 5}, MakeOp(batch)); 
    AT_DISPATCH_ALL_TYPES(edge_index -> scalar_type(), "combination", [&]{
        _combination<scalar_t><<<blkcmb, thcmb>>>(
                    pid -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                  batch -> packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                 num_edges.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
             edge_index -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                   num_pid.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     combi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                       msk.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     edge_dx, mx); 
    }); 
    combi = combi.index({msk > 0}); 
    msk   = msk.index({msk > 0}); 
    if (!msk.size({0})){return {};}

    double scale = (gev) ? 1.0/1000.0 : 1.0; 
    double mTu = mT * (2 - t_pm) * scale; 
    double mWu = mW * (2 - w_pm) * scale; 
    double mTl = mT * t_pm * scale; 
    double mWl = mW * w_pm * scale;
    double msW = (mWu - mWl) / double(steps);
    double msT = (mTu - mTl) / double(steps); 
    const unsigned int n_msk = msk.size({0}); 

    torch::Tensor mass  = torch::zeros({evnt_dx, 4}, MakeOp(pmc)); 
    torch::Tensor nu1   = torch::zeros({evnt_dx, 3}, MakeOp(pmc)); 
    torch::Tensor nu2   = torch::zeros({evnt_dx, 3}, MakeOp(pmc)); 
    torch::Tensor sol   = torch::zeros({evnt_dx}   , MakeOp(pmc)); 
    torch::Tensor edge  = torch::zeros({evnt_dx, 4}, MakeOp(batch)); 

    long dx_s = (steps+1); 
    const dim3 thmss  = dim3(1, 1); 
    const dim3 blkmss = blk_(dx_s, 1, dx_s, 1);
    torch::Tensor massTW = torch::ones({dx_s*dx_s, 2}, MakeOp(pmc)); 
    AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "massx", [&]{
        _mass_matrix<<<blkmss, thmss>>>(
                massTW.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                mTl, msT, mWl, msW, dx_s); 
    }); 

    torch::Tensor n1 = msk == 1; 
    torch::Tensor n2 = msk == 2; 
    torch::Tensor l1 = combi.index({torch::indexing::Slice(), 0}); 
    torch::Tensor l2 = combi.index({torch::indexing::Slice(), 1}); 
    torch::Tensor b1 = combi.index({torch::indexing::Slice(), 2}); 
    torch::Tensor b2 = combi.index({torch::indexing::Slice(), 3}); 
    torch::Tensor ev = combi.index({torch::indexing::Slice(), 4}); 
    ev = ev.index({(torch::ones({n_msk, dx_s * dx_s}, MakeOp(batch)).cumsum({0})-1).view({-1})});

    torch::Tensor pmcl1 = pmc -> index({l1, true})*scale; 
    torch::Tensor pmcl2 = pmc -> index({l2, true})*scale; 
    torch::Tensor pmcb1 = pmc -> index({b1, true})*scale; 
    torch::Tensor pmcb2 = pmc -> index({b2, true})*scale; 
    torch::Tensor metxy = met_xy -> index({ev, true})*scale; 

    std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(&pmcb1, &pmcl1, &massTW); 
    torch::Tensor H1_inv = std::get<0>(operators_::Inverse(&H1_m["H_perp"])); 
    torch::Tensor H1     = H1_m["H"]; 

    std::map<std::string, torch::Tensor> H2_m = nusol_::BaseMatrix(&pmcb2, &pmcl2, &massTW); 
    torch::Tensor H2_inv = std::get<0>(operators_::Inverse(&H2_m["H_perp"])); 
    torch::Tensor H2     = H2_m["H"]; 
   
    torch::Tensor tmp = massTW.clone();  
    for (size_t x(0); x < n_msk-1; ++x){massTW = torch::cat({massTW, tmp}, {0});}

    std::map<std::string, torch::Tensor> solx = nusol_::NuNu(&H1, &H1_inv, &H2, &H2_inv, &metxy, null); 
    torch::Tensor solxT = solx["distances"].reshape({-1, n_msk, 18, 1}); 
    torch::Tensor nu1_  = solx["nu1"].reshape({-1, n_msk, 18, 3});
    torch::Tensor nu2_  = solx["nu2"].reshape({-1, n_msk, 18, 3});
    torch::Tensor mTW   = massTW.reshape({-1, n_msk, 2});
    torch::Tensor evn   = ev.reshape({-1, n_msk}); 
    unsigned int mxsolx = mTW.size({0}); 

    const dim3 thsol  = dim3(n_msk); 
    const dim3 blksol = blk_(n_msk, n_msk);
    const unsigned int size_sol = sizeof(double)*(n_msk*n_msk + evnt_dx * evnt_dx + evnt_dx)*6; 
    AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "compare_solx", [&]{
        _compare_solx<scalar_t><<<blksol, thsol, size_sol>>>(
                    evn.packed_accessor64<long  , 2, torch::RestrictPtrTraits>(),
                    sol.packed_accessor64<double, 1, torch::RestrictPtrTraits>(),
                   mass.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                   edge.packed_accessor64<long  , 2, torch::RestrictPtrTraits>(),
                    nu1.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    nu2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),

                    mTW.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                  combi.packed_accessor64<long  , 2, torch::RestrictPtrTraits>(),
                  solxT.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),
                   nu1_.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),
                   nu2_.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),
                    n_msk, mxsolx, evnt_dx); 
    }); 
    
    std::map<std::string, torch::Tensor> out; 
    out["l1"] = edge.index({torch::indexing::Slice(), 0, true}); 
    out["l2"] = edge.index({torch::indexing::Slice(), 1, true}); 
    out["b1"] = edge.index({torch::indexing::Slice(), 2, true}); 
    out["b2"] = edge.index({torch::indexing::Slice(), 3, true}); 
    out["distances"] = sol; 
    out["masses"] = mass; 
    out["nu1"] = nu1;
    out["nu2"] = nu2; 
    out["msk"] = msk; 
    return out; 
}


