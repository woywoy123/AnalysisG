#include <nusol/nusol.cuh>
#include <nusol/base.cuh>
#include <nusol/utils.cuh>
#include <utils/utils.cuh>
#include <physics/physics.cuh>

std::map<std::string, torch::Tensor> nusol_::combinatorial(
    torch::Tensor* edge_index, torch::Tensor* batch, torch::Tensor* pmc, 
    torch::Tensor* pid, torch::Tensor* met_xy, 
    double  mT, double mW, double null, double perturb, long steps, bool gev
){
    const double   scale = (gev) ? 1.0 : 1.0/1000.0; 
    const double _mass_t = scale * mT; 
    const double _mass_w = scale * mW; 

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

    // build matrix
    // -------------- output ----------- //
    torch::Tensor nu1   = torch::zeros({evnt_dx, 4}, MakeOp(pmc)); 
    torch::Tensor nu2   = torch::zeros({evnt_dx, 4}, MakeOp(pmc)); 
    torch::Tensor sol   = torch::zeros({evnt_dx   }, MakeOp(pmc)); 
    torch::Tensor edge  = torch::zeros({evnt_dx, 4}, MakeOp(batch)); 
    // --------------------------------- //

    torch::Tensor n1 = (msk == 1); 
    torch::Tensor n2 = (msk == 2); 
    torch::Tensor l1 = combi.index({torch::indexing::Slice(), 0}); 
    torch::Tensor l2 = combi.index({torch::indexing::Slice(), 1}); 
    torch::Tensor b1 = combi.index({torch::indexing::Slice(), 2}); 
    torch::Tensor b2 = combi.index({torch::indexing::Slice(), 3}); 
    torch::Tensor ev = combi.index({torch::indexing::Slice(), 4}); 
    torch::Tensor metxy = met_xy -> index({ev})*scale; 
    torch::Tensor ev_id = (torch::ones({combi.size({0})},  MakeOp(batch)).cumsum({0})-1).view({-1}); 

    const unsigned int lenx = ev_id.size({0}); 
    torch::Tensor nu1_    = torch::zeros({lenx, 6, 3}, MakeOp(pmc));
    torch::Tensor nu2_    = torch::zeros({lenx, 6, 3}, MakeOp(pmc));
    torch::Tensor dst_    = torch::zeros({lenx, 6   }, MakeOp(pmc));
    torch::Tensor mass_tw = torch::zeros({lenx, 2   }, MakeOp(pmc)); 

    const dim3 thdm = dim3(lenx, 2);
    const dim3 bldm = blk_(lenx, lenx, 2, 2); 
    AT_DISPATCH_ALL_TYPES(edge_index -> scalar_type(), "assign_mass", [&]{
        _assign_mass<<<bldm, thdm>>>(
                mass_tw.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                _mass_t, _mass_w, lenx
        );
    }); 
    std::map<std::string, torch::Tensor> nus; 
    if (n1.index({n1}).size({0})){
        return {}; 
        unsigned int sqp = 7; 
        torch::Tensor _msk = n1.index({ev_id}); 
        unsigned int lx = _msk.index({_msk}).size({0});

        torch::Tensor idx = (torch::ones({lx, sqp  }, MakeOp(batch)).cumsum({0})-1).reshape({-1});
        torch::Tensor idp = (torch::ones({lx, sqp*6}, MakeOp(batch)).cumsum({0})-1).reshape({-1});

        torch::Tensor _pmcl1 = (pmc -> index({l2})*scale).index({n1}).index({idx}).clone(); 
        torch::Tensor _pmcb1 = (pmc -> index({b1})*scale).index({n1}).index({idx}).clone(); 

        torch::Tensor dnu_met =   metxy.index({_msk}).index({idx}).clone(); 
        torch::Tensor dnu_tw1 = mass_tw.index({_msk}).index({idx}).clone(); 
        torch::Tensor dnu_tw2 = mass_tw.index({_msk}).index({idx}).clone(); 
        torch::Tensor params  = torch::cat({dnu_met, dnu_tw1, dnu_tw2}, {-1}); 
        torch::Tensor ones  = torch::ones_like(dnu_tw1); 

        const unsigned long thx = (lx > 24) ? 24 : lx; 
        const dim3 thdx  = dim3(thx    , sqp     , 6   ); 
        const dim3 blkdx = blk_(lx, thx, sqp, sqp, 6, 6);

        const dim3 thdJx = dim3(thx    , sqp-1       ); 
        const dim3 blkJx = blk_(lx, thx, sqp-1, sqp-1);

        long tmp_s = 0; 
        double tmp_p = perturb * 1000.0 * scale; 

        torch::Tensor tx = (_pmcb1 + _pmcl1); 
        tx -= torch::cat({dnu_met, torch::zeros_like(dnu_met)}, {-1}); 
        tx  = tx.index({idp, torch::indexing::Slice(0, 3)});
        for (size_t x(0); x < steps; ++x){
            AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "perturb", [&]{
                _perturb<24, 6><<<blkdx, thdx>>>(
                     params.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    dnu_met.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                    dnu_tw1.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                    dnu_tw2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                    lx, perturb, sqp
                ); 
            });

            nus = nusol_::Nu(&_pmcb1, &_pmcl1, &dnu_met, &dnu_tw1, nullptr, null); 
            torch::Tensor dnu_res = nus["distances"];
            torch::Tensor solx = (dnu_res != 0).sum({-1});
            torch::Tensor ntx = torch::pow(tx + (nus["nu"].view({-1, 3})), 2).sum({-1}).view({-1, 6}); 
            dnu_res = torch::pow(torch::abs(ntx + dnu_res).sum({-1}, true) / (solx.view({-1, 1})+1), 0.5); 

            solx = (solx*nus["passed"] != 0);
            bool nox = (solx.index({solx}).size({0})) == 0; 
            AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "minimize", [&]{
                _jacobi<24, 6><<<blkJx, thdJx>>>(
                           params.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                          dnu_res.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                              lx, perturb, sqp, x
                ); 
            });

            nus["distances"] = dnu_res; 
            std::cout << nus["nu"] << std::endl; 
            perturb = (nox) ? (0.5 - x%2)*(1 + rand() % 100)*tmp_p : tmp_p; 
            if (!nox){tmp_s = 0; continue;}
            params = torch::cat({dnu_met, dnu_tw1 + ones*perturb, dnu_tw2 + ones*perturb}, {-1});
            if (tmp_s){break;}
            tmp_s = 100; 
        }
        idx = (torch::ones({lx, sqp}, MakeOp(batch)).cumsum({1}) - sqp).reshape({-1}) == 0;
        dst_.index_put_({_msk}, nus["distances"].index({idx})); 
        nu1_.index_put_({_msk}, nus["nu"].index({idx})); 
        nu2_.index_put_({_msk}, torch::zeros_like(nus["nu"]).index({idx})); 
    }

    if (n2.index({n2}).size({0})){
        std::map<std::string, torch::Tensor> nus_; 
        torch::Tensor _msk = n2.index({ev_id}); 
        double tmp_p = perturb * 1000.0 * scale; 

        long tmp_s = 1000;  
        const unsigned int mxi = 12; 
        const unsigned int sqp = 8; 
        const unsigned int sqx = sqp-1; 
        const unsigned int lx = _msk.index({_msk}).size({0});
        const unsigned long thx = (lx > mxi) ? mxi : lx; 

        torch::Tensor idx = (torch::ones({lx, sqp  }, MakeOp(batch)).cumsum({0})-1).reshape({-1});
        torch::Tensor idp = (torch::ones({lx, sqp*6}, MakeOp(batch)).cumsum({0})-1).reshape({-1});
        torch::Tensor oxn = metxy.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)}); 
        metxy = torch::cat({metxy, torch::zeros_like(oxn)}, {-1});

        torch::Tensor _pmcl1  = (pmc -> index({l1})*scale).index({n2}).index({idx}).clone(); 
        torch::Tensor _pmcb1  = (pmc -> index({b1})*scale).index({n2}).index({idx}).clone(); 
        torch::Tensor _pmcl2  = (pmc -> index({l2})*scale).index({n2}).index({idx}).clone(); 
        torch::Tensor _pmcb2  = (pmc -> index({b2})*scale).index({n2}).index({idx}).clone(); 

        // paras: met_x, met_y, met_z, mt1, mt2, mw1, mw2 = 7 -> 7 x 7
        torch::Tensor dnu_met =   metxy.index({_msk}).index({idx}).clone(); 
        torch::Tensor dnu_tw1 = mass_tw.index({_msk}).index({idx}).clone(); 
        torch::Tensor dnu_tw2 = mass_tw.index({_msk}).index({idx}).clone(); 
        torch::Tensor params  = torch::cat({metxy, mass_tw, mass_tw}, {-1}).index({_msk}).clone(); 

        const dim3 thdx  = dim3(thx    , sqp     , sqx     ); 
        const dim3 blkdx = blk_(lx, thx, sqp, sqp, sqx, sqx);

        const dim3 thdJx = dim3(thx    , sqx     ); 
        const dim3 blkJx = blk_(lx, thx, sqx, sqx);

        torch::Tensor tx = torch::cat({dnu_met, oxn.index({_msk}).index({idx})}, {-1}); 
        torch::Tensor pmx  = (_pmcb1 + _pmcl1 + _pmcb2 + _pmcl2).index({idp, torch::indexing::Slice(0, 3)});
        
        idx = (torch::ones({lx*sqp}, MakeOp(batch)).cumsum({0})-1).view({-1, 1});
        for (size_t x(0); x < steps; ++x){
            AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "perturb", [&]{
                _perturb<mxi, sqx><<<blkdx, thdx>>>(
                     params.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    dnu_met.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                    dnu_tw1.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                    dnu_tw2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                    lx, tmp_p, sqp
                ); 
            });


            nus = nusol_::NuNu(&_pmcb1, &_pmcb2, &_pmcl1, &_pmcl2, &dnu_met, 1e8, &dnu_tw1, &dnu_tw2, 1e-6, 1e-4, tmp_s);
            torch::Tensor dnu_res = nus["distances"];
            torch::Tensor solx = (dnu_res != 0).sum({-1});
            bool nox = (solx.index({solx}).size({0})) == 0; 

            torch::Tensor mtw = torch::pow(nus["mtw"].view({-1, 4}) - torch::cat({dnu_tw1, dnu_tw2}, {-1}).index({idp}), 2).sum({-1}, true).view({-1, 6});
            nus["distances"] = (dnu_res + mtw) * (nus["distances"] != 0); 

            torch::Tensor ntx = (nus["nu1"] + nus["nu2"]).view({-1, 3}); 
            dnu_res = pow((pmx + dnu_met.index({idp}) + ntx), 2).sum({-1}, true).view({-1, 6}) + mtw;

            AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "minimize", [&]{
                _jacobi<mxi, sqx><<<blkJx, thdJx>>>(
                           params.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
                          dnu_res.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                              lx, tmp_p, sqp, x
                ); 
            });

            if (!nus_.size()){nus_ = nus;}
            if (nox){tmp_s = 100; continue;}
            nus_["distances"] = torch::cat({nus_["distances"], nus["distances"]}, {-1}); 
            nus_["distances"] += (nus_["distances"] == 0)*1e8;
            nus_["nu1"] = torch::cat({nus_["nu1"], nus["nu1"]}, {-2}); 
            nus_["nu2"] = torch::cat({nus_["nu2"], nus["nu2"]}, {-2}); 
            nus_["mtw"] = torch::cat({nus_["mtw"], nus["mtw"]}, {-2}); 

            std::tuple<torch::Tensor, torch::Tensor> mx = torch::sort(nus_["distances"]); 
            nus_["distances"] = nus_["distances"].gather(1, std::get<1>(mx)).index({torch::indexing::Slice(), torch::indexing::Slice(0, 6)}); 
            nus_["nu1"] = nus_["nu1"].index({idx, std::get<1>(mx)}).index({torch::indexing::Slice(), torch::indexing::Slice(0, 6)}); 
            nus_["nu2"] = nus_["nu2"].index({idx, std::get<1>(mx)}).index({torch::indexing::Slice(), torch::indexing::Slice(0, 6)});
            nus_["mtw"] = nus_["mtw"].index({idx, std::get<1>(mx)}).index({torch::indexing::Slice(), torch::indexing::Slice(0, 6)});
            nus_["distances"] *= (nus_["distances"] != 1e8); 
            tmp_s = 0; 
        }
        idx = (torch::ones({lx, sqp}, MakeOp(batch)).cumsum({1}) - sqp).reshape({-1}) == 0;
        dst_.index_put_({_msk}, nus_["distances"].index({idx})); 
        nu1_.index_put_({_msk}, nus_["nu1"].index({idx})); 
        nu2_.index_put_({_msk}, nus_["nu2"].index({idx})); 
    }

    const unsigned long thx = (lenx > 32) ? 32 : lenx; 
    const dim3 thsol  = dim3(thx, 4); 
    const dim3 blksol = blk_(evnt_dx, thx, 4, 4);
    AT_DISPATCH_ALL_TYPES(pmc -> scalar_type(), "compare_solx", [&]{
        _compare_solx<32><<<blksol, thsol>>>(
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
    out["nu1"] = nu1 * (1.0 / scale);
    out["nu2"] = nu2 * (1.0 / scale); 
    out["msk"] = msk; 

    out["alt.nu1"] = nu1_ * (1.0 / scale); 
    out["alt.nu2"] = nu2_ * (1.0 / scale); 
    out["alt.l1"] = combi.index({torch::indexing::Slice(), 0, true});
    out["alt.l2"] = combi.index({torch::indexing::Slice(), 1, true});
    out["alt.b1"] = combi.index({torch::indexing::Slice(), 2, true});
    out["alt.b2"] = combi.index({torch::indexing::Slice(), 3, true});
    out["alt.distances"] = dst_; 
    return out; 
}


