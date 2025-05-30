#include <transform/transform.cuh>
#include <transform/base.cuh>
#include <utils/utils.cuh>

#ifndef phys_th
#define phys_th 128
#endif

torch::Tensor transform_::Px(torch::Tensor* pt, torch::Tensor* phi){
    const unsigned int dx = pt -> size(0); 
    torch::Tensor px_  = torch::zeros({dx, 1}, MakeOp(pt)); 
    torch::Tensor pt_  =  pt -> reshape({-1, 1});  
    torch::Tensor phi_ = phi -> reshape({-1, 1});  

    const dim3 thd = dim3(1); 
    const dim3 blk = blk_(dx, 1);

    AT_DISPATCH_FLOATING_TYPES(pt -> scalar_type(), "px", [&]{
        PxK<scalar_t><<<blk, thd>>>(
               pt_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
              phi_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               px_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }); 
    return px_; 
}

torch::Tensor transform_::Py(torch::Tensor* pt, torch::Tensor* phi){
    const unsigned int dx = pt -> size(0); 
    torch::Tensor py_  = torch::zeros({dx, 1}, MakeOp(pt)); 
    torch::Tensor pt_  =  pt -> reshape({-1, 1});  
    torch::Tensor phi_ = phi -> reshape({-1, 1});  

    const dim3 thd = dim3(1); 
    const dim3 blk = blk_(dx, 1);

    AT_DISPATCH_FLOATING_TYPES(pt -> scalar_type(), "py", [&]{
        PyK<scalar_t><<<blk, thd>>>(
               pt_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
              phi_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               py_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    }); 
    return py_; 
}

torch::Tensor transform_::Pz(torch::Tensor* pt, torch::Tensor* eta){
    const unsigned int dx = pt -> size(0); 
    torch::Tensor pz_  = torch::zeros({dx, 1}, MakeOp(pt)); 
    torch::Tensor pt_  =  pt -> reshape({-1, 1});  
    torch::Tensor eta_ = eta -> reshape({-1, 1});  

    const dim3 thd = dim3(1); 
    const dim3 blk = blk_(dx, 1);

    AT_DISPATCH_FLOATING_TYPES(pt -> scalar_type(), "pz", [&]{
        PzK<scalar_t><<<blk, thd>>>(
               pt_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
              eta_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               pz_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    }); 
    return pz_; 
}

torch::Tensor transform_::PxPyPz(torch::Tensor* pmu){
    const unsigned int dx = pmu -> size({0}); 
    const unsigned int dy = pmu -> size({-1});
    torch::Tensor pmc = torch::zeros({dx, dy}, MakeOp(pmu)); 

    const unsigned int thx = (dx >= phys_th) ? phys_th : dx; 
    const dim3 thd = dim3(thx, 4); 
    const dim3 blk = blk_(dx, thx, 4, 4);

    AT_DISPATCH_FLOATING_TYPES(pmu -> scalar_type(), "pxpypz", [&]{ 
        PxPyPzK<scalar_t, phys_th><<<blk, thd>>>(
             pmu -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                dx, dy);
    }); 
    return pmc;   
}

torch::Tensor transform_::PxPyPzE(torch::Tensor* pmu){
    const unsigned int dx = pmu -> size({0}); 
    const unsigned int dy = pmu -> size({-1});
    if (dy >= 4){return transform_::PxPyPz(pmu);}
    torch::Tensor pmc = torch::zeros({dx, dy}, MakeOp(pmu)); 

    const unsigned int thx = (dx >= phys_th) ? phys_th : dx; 
    const dim3 thd = dim3(thx, 4); 
    const dim3 blk = blk_(dx, thx, 4, 4);
    AT_DISPATCH_FLOATING_TYPES(pmu -> scalar_type(), "pxpypze", [&]{ 
        PxPyPzEK<scalar_t, phys_th><<<blk, thd>>>(
             pmu -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                dx, dy);
    }); 
    return pmc;   
}

torch::Tensor transform_::Pt(torch::Tensor* px, torch::Tensor* py){
    torch::Tensor px_ = px -> reshape({-1, 1}); 
    torch::Tensor py_ = py -> reshape({-1, 1});    

    const unsigned int dx = px -> size({0}); 
    const dim3 thd = dim3(1); 
    const dim3 blk = blk_(dx, 1);
    torch::Tensor pt_ = torch::zeros({dx, 1}, MakeOp(px)); 

    AT_DISPATCH_FLOATING_TYPES(pt_.scalar_type(), "pt", [&]{
        PtK<scalar_t><<<blk, thd>>>(
            px_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pt_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }); 
    return pt_; 
}

torch::Tensor transform_::Phi(torch::Tensor* px, torch::Tensor* py){
    torch::Tensor px_ = px -> reshape({-1, 1}); 
    torch::Tensor py_ = py -> reshape({-1, 1});    
    torch::Tensor phi_ = torch::zeros_like(py_); 

    const unsigned int dx = px_.size(0); 
    const dim3 thd = dim3(1); 
    const dim3 blk = blk_(dx, 1);

    AT_DISPATCH_FLOATING_TYPES(phi_.scalar_type(), "phi", [&]{
        PhiK<scalar_t><<<blk, thd>>>(
            px_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           phi_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }); 
    return phi_; 
}


torch::Tensor transform_::Eta(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size(0); 
    const dim3 thd = dim3(1); 
    const dim3 blk = blk_(dx, 1);

    torch::Tensor eta = torch::zeros({dx, 1}, MakeOp(pmc)); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "eta", [&]{
        EtaK<scalar_t><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               eta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
    }); 
    return eta; 
}

torch::Tensor transform_::PtEtaPhi(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = pmc -> size({-1});
    torch::Tensor pmu = torch::zeros({dx, dy}, MakeOp(pmc)); 

    const unsigned int thx = (dx >= phys_th) ? phys_th : dx; 
    const dim3 thd = dim3(thx, 4); 
    const dim3 blk = blk_(dx, thx, 4, 4);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "ptetaphi", [&]{
        PtEtaPhiK<scalar_t, phys_th><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               pmu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx, dy
        ); 
    }); 
    return pmu; 
}

torch::Tensor transform_::PtEtaPhiE(torch::Tensor* pmc){
    const unsigned int dy = pmc -> size({-1});
    const unsigned int dx = pmc -> size({0}); 
    if (dy < 4){return transform_::PtEtaPhi(pmc);}
    torch::Tensor pmu = torch::zeros({dx, 4}, MakeOp(pmc)); 

    const unsigned int thx = (dx >= phys_th) ? phys_th : dx; 
    const dim3 thd = dim3(thx, 4); 
    const dim3 blk = blk_(dx, thx, 4, 4);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "ptetaphie", [&]{
        PtEtaPhiEK<scalar_t, phys_th><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               pmu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx, dy);
    }); 
    return pmu;   
}

torch::Tensor transform_::Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmc = format({*px, *py, *pz}); 
    return transform_::Eta(&pmc); 
}

torch::Tensor transform_::PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmc = format({*px, *py, *pz}); 
    return transform_::PtEtaPhi(&pmc);   
}

torch::Tensor transform_::PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmc = format({*px, *py, *pz, *e}); 
    return transform_::PtEtaPhi(&pmc);   
}

torch::Tensor transform_::PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi){
    torch::Tensor pmu = format({*pt, *eta, *phi}); 
    return transform_::PxPyPz(&pmu); 
}

torch::Tensor transform_::PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy){
    torch::Tensor pmu = format({*pt, *eta, *phi, *energy}); 
    return transform_::PxPyPz(&pmu); 
}


