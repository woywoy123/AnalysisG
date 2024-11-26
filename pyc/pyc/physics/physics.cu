#include <cutils/utils.cuh>
#include <transform/transform.cuh>
#include <physics/physics.cuh>
#include <physics/base.cuh>

torch::Tensor physics_::P2(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 3;
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor p2 = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1, dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "P2", [&]{ 
        _P2K<scalar_t><<<blk, thd, s>>>(
             pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                 p2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                 dy);  
    }); 
    return p2;  
}

torch::Tensor physics_::P(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 3; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor p = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "P", [&]{ 
        _PK<scalar_t><<<blk, thd, s>>>(
             pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                  p.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                  dy);  
    }); 
    return p;  
}

torch::Tensor physics_::Beta2(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 4; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor b2 = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "Beta2", [&]{ 
        _Beta2<scalar_t><<<blk, thd, s>>>(
                pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                    b2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());  
    }); 
    return b2;  
}

torch::Tensor physics_::Beta(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 4; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor b = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "Beta", [&]{ 
        _Beta<scalar_t><<<blk, thd, s>>>(
                pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                     b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());  
    }); 
    return b;  
}

torch::Tensor physics_::M2(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 4; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor m2 = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "M2", [&]{ 
        _M2<scalar_t><<<blk, thd, s>>>(
                pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                    m2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());  
    }); 
    return m2;  
}

torch::Tensor physics_::M(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 4; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor m = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "M", [&]{ 
        _M<scalar_t><<<blk, thd, s>>>(
                pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                     m.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());  
    }); 
    return m;  
}

torch::Tensor physics_::Mt2(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 4; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor mt2 = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "Mt2", [&]{ 
        _Mt2<scalar_t><<<blk, thd, s>>>(
                pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                   mt2.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());  
    }); 
    return mt2;  
}

torch::Tensor physics_::Mt(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 4; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor mt = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "Mt", [&]{ 
        _Mt<scalar_t><<<blk, thd, s>>>(
               pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                   mt.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());  
    }); 
    return mt;  
}

torch::Tensor physics_::Theta(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = 4; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor theta = torch::zeros({dx, 1}, MakeOp(pmc)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "theta", [&]{ 
        _theta<scalar_t><<<blk, thd, s>>>(
               pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                theta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());  
    }); 
    return theta;
}

torch::Tensor physics_::DeltaR(torch::Tensor* pmu1, torch::Tensor* pmu2){
    const unsigned int dx = pmu1 -> size({0}); 
    const unsigned int dy = 4; 
    const unsigned int s = sizeof(double)*dy; 
    torch::Tensor dr = torch::zeros({dx, 1}, MakeOp(pmu1)); 

    const dim3 thd = dim3(1 , dy); 
    const dim3 blk = blk_(dx, 1, dy, dy);

    AT_DISPATCH_FLOATING_TYPES(pmu1 -> scalar_type(), "deltar", [&]{ 
        _deltar<scalar_t><<<blk, thd, s>>>(
              pmu1 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
              pmu2 -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                   dr.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());  
    }); 
    return dr;  
}

torch::Tensor physics_::DeltaR(torch::Tensor* eta1, torch::Tensor* eta2, torch::Tensor* phi1, torch::Tensor* phi2){
    torch::Tensor pmu1 = format({*eta1, *eta1, *phi1, *eta1}); 
    torch::Tensor pmu2 = format({*eta2, *eta2, *phi2, *eta2}); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor physics_::P2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmx = format({*px, *py, *pz}); 
    return physics_::P2(&pmx);
}

torch::Tensor physics_::P(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmx = format({*px, *py, *pz}); 
    return physics_::P(&pmx); 
}

torch::Tensor physics_::Beta2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*px, *py, *pz, *e}); 
    return physics_::Beta2(&pmx);  
}

torch::Tensor physics_::Beta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*px, *py, *pz, *e}); 
    return physics_::Beta(&pmx);  
}

torch::Tensor physics_::M2(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*px, *py, *pz, *e}); 
    return physics_::M2(&pmx); 
}

torch::Tensor physics_::M(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*px, *py, *pz, *e}); 
    return physics_::M(&pmx);
}

torch::Tensor physics_::Mt2(torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*pz, *pz, *pz, *e}); 
    return physics_::Mt2(&pmx);
}

torch::Tensor physics_::Mt(torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmx = format({*pz, *pz, *pz, *e}); 
    return physics_::Mt(&pmx);
}


torch::Tensor physics_::Theta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmx = format({*px, *py, *pz}); 
    return physics_::Theta(&pmx); 
}


























