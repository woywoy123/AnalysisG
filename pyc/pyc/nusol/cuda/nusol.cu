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

    // build matrix
    long dx_s = steps; 
    const dim3 thmss  = dim3((dx_s >= 512) ? 512 : dx_s, 2); 
    const dim3 blkmss = blk_(dx_s, (dx_s >= 512) ? 512 : dx_s, 2, 2);
    torch::Tensor mass1TW = torch::rand({dx_s, 2}, MakeOp(pmc)); 
    torch::Tensor mass2TW = torch::rand({dx_s, 2}, MakeOp(pmc)); 

    AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "massx", [&]{
        _mass_matrix<<<blkmss, thmss>>>(mass1TW.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), mTl, msT, mWl, msW, dx_s); 
        _mass_matrix<<<blkmss, thmss>>>(mass2TW.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), mTl, msT, mWl, msW, dx_s); 
    }); 
    
    // -------------- output ----------- //
    torch::Tensor nu1   = torch::zeros({evnt_dx, 4}, MakeOp(pmc)); 
    torch::Tensor nu2   = torch::zeros({evnt_dx, 4}, MakeOp(pmc)); 
    torch::Tensor sol   = torch::zeros({evnt_dx   }, MakeOp(pmc)); 
    torch::Tensor edge  = torch::zeros({evnt_dx, 4}, MakeOp(batch)); 
    // --------------------------------- //

    torch::Tensor ev_id = (torch::ones({combi.size({0}), dx_s},  MakeOp(batch)).cumsum({0})-1).view({-1}); 

    torch::Tensor n1 = (msk == 1); 
    torch::Tensor n2 = (msk == 2); 
    torch::Tensor l1 = combi.index({torch::indexing::Slice(), 0}); 
    torch::Tensor l2 = combi.index({torch::indexing::Slice(), 1}); 
    torch::Tensor b1 = combi.index({torch::indexing::Slice(), 2}); 
    torch::Tensor b2 = combi.index({torch::indexing::Slice(), 3}); 
    torch::Tensor ev = combi.index({torch::indexing::Slice(), 4}); 
    torch::Tensor metxy = met_xy -> index({ev})*scale; 

    const unsigned int lenx = ev_id.size({0}); 
    torch::Tensor nu1_ = torch::zeros({lenx, 6, 3}, MakeOp(pmc));
    torch::Tensor nu2_ = torch::zeros({lenx, 6, 3}, MakeOp(pmc));
    torch::Tensor dst_ = torch::zeros({lenx, 6}, MakeOp(pmc));
    torch::Tensor msk_ = torch::cat({n1.index({ev_id}).view({-1, 1}), n2.index({ev_id}).view({-1, 1})}, -1); 

    if (n1.index({n1}).size({0})){
        std::map<std::string, torch::Tensor> s_nu; 
        torch::Tensor snu_metxy = metxy.index({ev_id}).index({n1.index({ev_id})}); 
        torch::Tensor snu_pmcl1 = (pmc -> index({l2})*scale).index({n1}); 
        torch::Tensor snu_pmcb1 = (pmc -> index({b1})*scale).index({n1}); 
        std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(&snu_pmcb1, &snu_pmcl1, &mass1TW); 
        s_nu = nusol_::Nu(&H1_m["H"], nullptr, &snu_metxy, null); 

        torch::Tensor nux = n1.index({ev_id}); 
        nu1_.index_put_({nux}, s_nu["nu"]); 
        dst_.index_put_({nux}, s_nu["distances"]); 
    }

    if (n2.index({n2}).size({0})){
        std::map<std::string, torch::Tensor> d_nu; 
        torch::Tensor dnu_metxy = (metxy.index({ev_id})).index({n2.index({ev_id})}); 
        torch::Tensor dnu_pmcl1 = (pmc -> index({l1})*scale).index({n2}); 
        torch::Tensor dnu_pmcl2 = (pmc -> index({l2})*scale).index({n2}); 
        torch::Tensor dnu_pmcb1 = (pmc -> index({b1})*scale).index({n2}); 
        torch::Tensor dnu_pmcb2 = (pmc -> index({b2})*scale).index({n2}); 
        d_nu = nusol_::NuNu(&dnu_pmcb1, &dnu_pmcb2, &dnu_pmcl1, &dnu_pmcl2, &dnu_metxy, null, &mass1TW, &mass2TW); 
        torch::Tensor nux = n2.index({ev_id}); 
        nu1_.index_put_({nux}, d_nu["nu1"]); 
        nu2_.index_put_({nux}, d_nu["nu2"]); 
        dst_.index_put_({nux}, d_nu["distances"]); 
    }

    const dim3 thsol  = dim3(32, 4); 
    const dim3 blksol = blk_(evnt_dx, 32, 4, 4);
    AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "compare_solx", [&]{
        _compare_solx<<<blksol, thsol>>>(
                  ev_id.packed_accessor64<long  , 1, torch::RestrictPtrTraits>(),
                     ev.packed_accessor64<long  , 1, torch::RestrictPtrTraits>(),

                    sol.packed_accessor64<double, 1, torch::RestrictPtrTraits>(),
                   edge.packed_accessor64<long  , 2, torch::RestrictPtrTraits>(),
                    nu1.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    nu2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),

                  combi.packed_accessor64<long  , 2, torch::RestrictPtrTraits>(),
                   dst_.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                   nu1_.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                   nu2_.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                    lenx, evnt_dx); 
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


