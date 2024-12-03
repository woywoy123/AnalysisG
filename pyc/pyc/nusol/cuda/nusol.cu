#include <nusol/nusol.cuh>
#include <nusol/base.cuh>
#include <nusol/utils.cuh>
#include <cutils/utils.cuh>
#include <operators/operators.cuh>

torch::Tensor reshaped(torch::Tensor* tx, torch::Tensor* ix, unsigned int dx, signed int dim){
    torch::Tensor id = ix -> index({torch::indexing::Slice(), dim});
    torch::Tensor  H = tx -> view({-1, dx, dx, 3, 3});
    H = H.index({torch::indexing::Slice(), id}).index({torch::indexing::Slice(), torch::indexing::Slice(), id});
    return H.view({-1, 3, 3}); 
}

torch::Tensor masked(torch::Tensor* tx, torch::Tensor* ix, unsigned int dx, signed int dim){
    torch::Tensor id = ix -> index({torch::indexing::Slice(), dim});
    torch::Tensor  H = tx -> view({-1, dx, dx});
    H = H.index({torch::indexing::Slice(), id}).index({torch::indexing::Slice(), torch::indexing::Slice(), id});
    return H.view({-1}); 
}

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
    msk   = msk.index({msk > 1}); 
    if (!msk.size({0})){return {};}

    double scale = (gev) ? 1.0/1000.0 : 1.0; 
    double mTu = mT * (2 - t_pm) * scale; 
    double mWu = mW * (2 - w_pm) * scale; 
    double mTl = mT * t_pm * scale; 
    double mWl = mW * w_pm * scale;
    double msW = (mWu - mWl) / double(steps);
    double msT = (mTu - mTl) / double(steps); 
    const unsigned int n_msk = msk.size({0}); 

    torch::Tensor nu1   = torch::zeros({evnt_dx, 3}, MakeOp(pmc)); 
    torch::Tensor nu2   = torch::zeros({evnt_dx, 3}, MakeOp(pmc)); 
    torch::Tensor sol   = torch::zeros({evnt_dx}   , MakeOp(pmc)); 
    torch::Tensor edge  = torch::zeros({evnt_dx, 4}, MakeOp(batch)); 

    torch::Tensor n1 = msk == 1; 
    torch::Tensor n2 = msk == 2; 
    torch::Tensor l1 = combi.index({torch::indexing::Slice(), 0}); 
    torch::Tensor l2 = combi.index({torch::indexing::Slice(), 1}); 
    torch::Tensor b1 = combi.index({torch::indexing::Slice(), 2}); 
    torch::Tensor b2 = combi.index({torch::indexing::Slice(), 3}); 
    torch::Tensor ev = combi.index({torch::indexing::Slice(), 4}); 

    // build matrix
    long dx_s = steps; 
    const dim3 thmss  = dim3(1, 1); 
    const dim3 blkmss = blk_(dx_s, 1, dx_s, 1);
    torch::Tensor massTW = torch::rand({dx_s*dx_s, 2}, MakeOp(pmc)); 

    const dim3 thmc   = dim3(_threads, 1); 
    const dim3 blkmc  = blk_(dx_s*dx_s, _threads, 2, 1);
    torch::Tensor ix = torch::zeros({dx_s*dx_s, 2}, MakeOp(batch)); 

    AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "massx", [&]{
        _mass_matrix<<<blkmss, thmss>>>(
                massTW.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), mTl, msT, mWl, msW, dx_s
        ); 
        _combination_matrix<<<blkmc, thmc>>>(
                ix.packed_accessor64<long, 2, torch::RestrictPtrTraits>(), dx_s
        ); 
    }); 

    torch::Tensor pmcl1 = pmc -> index({l1, true})*scale; 
    torch::Tensor pmcl2 = pmc -> index({l2, true})*scale; 
    torch::Tensor pmcb1 = pmc -> index({b1, true})*scale; 
    torch::Tensor pmcb2 = pmc -> index({b2, true})*scale; 
    torch::Tensor metxy = met_xy -> index({ev, true})*scale; 

    std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(&pmcb1, &pmcl1, &massTW); 
    torch::Tensor H1      = reshaped(&H1_m["H"]     , &ix, dx_s, 0); 
    torch::Tensor H1_inv  = reshaped(&H1_m["H_perp"], &ix, dx_s, 0); 
    H1_inv = std::get<0>(operators_::Inverse(&H1_inv)); 

    std::map<std::string, torch::Tensor> H2_m = nusol_::BaseMatrix(&pmcb2, &pmcl2, &massTW); 
    torch::Tensor H2      = reshaped(&H2_m["H"]     , &ix, dx_s, 1); 
    torch::Tensor H2_inv  = reshaped(&H2_m["H_perp"], &ix, dx_s, 1); 
    H2_inv = std::get<0>(operators_::Inverse(&H2_inv)); 
    torch::Tensor skp = (torch::abs(H1_inv) + torch::abs(H2_inv)).sum({-1}).sum({-1}) > 0; 
    
    H1_inv = H1_inv.index({skp}); 
    H2_inv = H2_inv.index({skp}); 
    H1 = H1.index({skp}); 
    H2 = H2.index({skp}); 

    torch::Tensor xt = (torch::ones({ev.size({0}), dx_s*dx_s*dx_s*dx_s}, MakeOp(batch)).cumsum({0})-1); 
    xt    = xt.view({-1}).index({skp}); 
    metxy = metxy.index({xt}); 
    ev    = ev.index({xt}); 

    std::map<std::string, torch::Tensor> solx = nusol_::NuNu(&H1, &H1_inv, &H2, &H2_inv, &metxy, null); 
    torch::Tensor cmx = (torch::ones({combi.size({0})}, MakeOp(batch)).cumsum({0})-1).index({xt}); 

    torch::Tensor nu1_ = solx["nu1"].reshape({-1, 3}); 
    torch::Tensor nu2_ = solx["nu2"].reshape({-1, 3});  
    torch::Tensor solD = solx["distances"].reshape({-1});  

    const unsigned int lenx = nu1_.size({0})/18; 
    const dim3 thsol  = dim3(evnt_dx, n_msk); 
    const dim3 blksol = blk_(evnt_dx, evnt_dx, n_msk, n_msk);
    const unsigned int size_sol = sizeof(double)*n_msk*evnt_dx*4;
    AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "compare_solx", [&]{
        _compare_solx<<<blksol, thsol, size_sol>>>(
                     ev.packed_accessor64<long  , 1, torch::RestrictPtrTraits>(),
                    sol.packed_accessor64<double, 1, torch::RestrictPtrTraits>(),
                   edge.packed_accessor64<long  , 2, torch::RestrictPtrTraits>(),
                    nu1.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    nu2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    cmx.packed_accessor64<long  , 1, torch::RestrictPtrTraits>(),

                  combi.packed_accessor64<long  , 2, torch::RestrictPtrTraits>(),
                   solD.packed_accessor64<double, 1, torch::RestrictPtrTraits>(),
                   nu1_.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                   nu2_.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    lenx,  n_msk, evnt_dx); 
    }); 

    std::map<std::string, torch::Tensor> out;
    out["distances"] = sol; 
    out["l1"] = edge.index({torch::indexing::Slice(), 0, true}); 
    out["l2"] = edge.index({torch::indexing::Slice(), 1, true}); 
    out["b1"] = edge.index({torch::indexing::Slice(), 2, true}); 
    out["b2"] = edge.index({torch::indexing::Slice(), 3, true}); 
    out["nu1"] = nu1;
    out["nu2"] = nu2; 
    out["msk"] = msk; 
    return out; 
}


