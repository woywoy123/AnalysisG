#include <cutils/utils.cuh>
#include <transform/transform.cuh>
#include <physics/physics.cuh>
#include <physics/base.cuh>

torch::Tensor physics_::P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmx = format({*px, *py, *pz}); 
    const unsigned int dx = pmx.size({0}); 
    const unsigned int dy = pmx.size({1}); 
    const dim3 thd = dim3(64); 
    const dim3 blk = blk_(dx*dy, 64);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "P2", [&]{ 
        _P2K<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx, dy); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::P2(torch::Tensor* pmc){
    torch::Tensor pmx = pmc -> clone(); 
    pmx = pmx.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3)}); 
    const unsigned int dx = pmx.size({0}); 
    const unsigned int dy = pmx.size({1}); 

    const dim3 thd = dim3(64); 
    const dim3 blk = blk_(dx*dy, 64);
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "P2", [&]{ 
        _P2K<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx, dy); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
     torch::Tensor pmx = format({*px, *py, *pz}); 
    const unsigned int dx = pmx.size({0}); 
    const unsigned int dy = pmx.size({1}); 
    const dim3 thd = dim3(64); 
    const dim3 blk = blk_(dx*dy, 64);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "P", [&]{ 
        _PK<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx, dy); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
   
}

torch::Tensor physics_::P(torch::Tensor* pmc){
    torch::Tensor pmx = pmc -> clone(); 
    pmx = pmx.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3)}); 
    const unsigned int dx = pmx.size({0}); 
    const unsigned int dy = pmx.size({1}); 
    const dim3 thd = dim3(64); 
    const dim3 blk = blk_(dx*dy, 64);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "P", [&]{ 
        _PK<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx, dy); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*px, *py, *pz, *e}); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 4); 
    const dim3 blk = blk_(dx, 64, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "Beta2", [&]{ 
        _Beta2<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Beta2(torch::Tensor* pmc){
    torch::Tensor pmx = pmc -> clone(); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 4); 
    const dim3 blk = blk_(dx, 64, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "Beta2", [&]{ 
        _Beta2<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*px, *py, *pz, *e}); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 4); 
    const dim3 blk = blk_(dx, 64, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "Beta", [&]{ 
        _Beta<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Beta(torch::Tensor* pmc){
    torch::Tensor pmx = pmc -> clone(); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 4); 
    const dim3 blk = blk_(dx, 64, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "Beta", [&]{ 
        _Beta<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::M2(torch::Tensor* pmc){
    torch::Tensor pmx = pmc -> clone(); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 4); 
    const dim3 blk = blk_(dx, 64, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "M2", [&]{ 
        _M2<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*px, *py, *pz, *e}); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 4); 
    const dim3 blk = blk_(dx, 64, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "M2", [&]{ 
        _M2<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::M(torch::Tensor* pmc){
    torch::Tensor pmx = pmc -> clone(); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 4); 
    const dim3 blk = blk_(dx, 64, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "M", [&]{ 
        _M<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*px, *py, *pz, *e}); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 4); 
    const dim3 blk = blk_(dx, 64, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "M", [&]{ 
        _M<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Mt2(torch::Tensor* pmc){
    torch::Tensor pmx = format({
            pmc -> index({torch::indexing::Slice(), 2, true}),
            pmc -> index({torch::indexing::Slice(), 3, true})
    });
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 2); 
    const dim3 blk = blk_(dx, 64, 2, 2);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "Mt2", [&]{ 
        _Mt2<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}


torch::Tensor physics_::Mt2(torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*pz, *e}); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 2); 
    const dim3 blk = blk_(dx, 64, 2, 2);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "Mt2", [&]{ 
        _Mt2<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Mt(torch::Tensor* pmc){
    torch::Tensor pmx = format({
            pmc -> index({torch::indexing::Slice(), 2, true}),
            pmc -> index({torch::indexing::Slice(), 3, true})
    });
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 2); 
    const dim3 blk = blk_(dx, 64, 2, 2);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "Mt", [&]{ 
        _Mt<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Mt(torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*pz, *e}); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 2); 
    const dim3 blk = blk_(dx, 64, 2, 2);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "Mt", [&]{ 
        _Mt<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Theta(torch::Tensor* pmc){
    torch::Tensor pmx = pmc -> clone(); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 3); 
    const dim3 blk = blk_(dx, 64, 3, 3);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "theta", [&]{ 
        _theta<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmx = format({*px, *py, *pz}); 
    const unsigned int dx = pmx.size({0}); 
    const dim3 thd = dim3(64, 3); 
    const dim3 blk = blk_(dx, 64, 3, 3);
    AT_DISPATCH_FLOATING_TYPES(pmx.scalar_type(), "theta", [&]{ 
        _theta<scalar_t><<<blk, thd>>>(pmx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), dx); 
    }); 
    return pmx.index({torch::indexing::Slice(), 0, true}); 
}

torch::Tensor physics_::DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2){
    torch::Tensor dr = torch::zeros_like(pmu1 -> index({torch::indexing::Slice(), 0, true})); 
    const unsigned int dx = pmu1 -> size({0}); 
    const dim3 thd = dim3(64, 2); 
    const dim3 blk = blk_(dx, 64, 2, 2);
    AT_DISPATCH_FLOATING_TYPES(dr.scalar_type(), "deltar", [&]{ 
        _deltar<scalar_t><<<blk, thd>>>(
                pmu1 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                pmu2 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                dr.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                dx); 
    }); 
    return dr; 
}

torch::Tensor physics_::DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2){
    torch::Tensor dr = torch::zeros_like(*eta1); 
    torch::Tensor pmu1 = format({*eta1, *phi1}); 
    torch::Tensor pmu2 = format({*eta2, *phi2}); 
    const unsigned int dx = eta1 -> size({0}); 
    const dim3 thd = dim3(64, 2); 
    const dim3 blk = blk_(dx, 64, 2, 2);
    AT_DISPATCH_FLOATING_TYPES(dr.scalar_type(), "deltar", [&]{ 
        _deltar<scalar_t><<<blk, thd>>>(
                pmu1.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                pmu2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                dr.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                dx, 0); 
    }); 
    return dr; 
}

