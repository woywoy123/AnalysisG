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

    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    AT_DISPATCH_FLOATING_TYPES(pt -> scalar_type(), "px", [&]{
        PxK<scalar_t><<<blk, thd>>>(
               pt_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
              phi_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               px_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx
        );
    }); 
    return px_; 
}

torch::Tensor transform_::Py(torch::Tensor* pt, torch::Tensor* phi){
    const unsigned int dx = pt -> size(0); 
    torch::Tensor py_  = torch::zeros({dx, 1}, MakeOp(pt)); 
    torch::Tensor pt_  =  pt -> reshape({-1, 1});  
    torch::Tensor phi_ = phi -> reshape({-1, 1});  

    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    AT_DISPATCH_FLOATING_TYPES(pt -> scalar_type(), "py", [&]{
        PyK<scalar_t><<<blk, thd>>>(
               pt_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
              phi_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               py_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx
        );
    }); 
    return py_; 
}

torch::Tensor transform_::Pz(torch::Tensor* energy, torch::Tensor* eta){
    const unsigned int dx = energy -> size(0); 
    torch::Tensor pz_  = torch::zeros({dx, 1}, MakeOp(energy)); 
    torch::Tensor en_  = energy -> reshape({-1, 1});  
    torch::Tensor eta_ = eta -> reshape({-1, 1});  

    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    AT_DISPATCH_FLOATING_TYPES(eta -> scalar_type(), "pz", [&]{
        PzK<scalar_t><<<blk, thd>>>(
               en_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
              eta_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               pz_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx
        );
    }); 
    return pz_; 
}



torch::Tensor transform_::Pt(torch::Tensor* px, torch::Tensor* py){
    torch::Tensor px_ = px -> reshape({-1, 1}); 
    torch::Tensor py_ = py -> reshape({-1, 1});    

    const unsigned int dx = px -> size({0}); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);
    torch::Tensor pt_ = torch::zeros({dx, 1}, MakeOp(px)); 

    AT_DISPATCH_FLOATING_TYPES(pt_.scalar_type(), "pt", [&]{
        PtK<scalar_t><<<blk, thd>>>(
            px_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pt_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    }); 
    return pt_; 
}

torch::Tensor transform_::Phi(torch::Tensor* px, torch::Tensor* py){
    torch::Tensor px_ = px -> reshape({-1, 1}); 
    torch::Tensor py_ = py -> reshape({-1, 1});    
    torch::Tensor phi_ = torch::zeros_like(py_); 

    const unsigned int dx = px_.size(0); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    AT_DISPATCH_FLOATING_TYPES(phi_.scalar_type(), "phi", [&]{
        PhiK<scalar_t><<<blk, thd>>>(
            px_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            py_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           phi_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           dx
        );
    }); 
    return phi_; 
}

torch::Tensor transform_::Eta(torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pz_ = pz -> reshape({-1, 1});    
    torch::Tensor  e_ =  e -> reshape({-1, 1}); 

    const unsigned int dx = pz -> size({0}); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);
    torch::Tensor eta_ = torch::zeros({dx, 1}, MakeOp(e)); 

    AT_DISPATCH_FLOATING_TYPES(eta_.scalar_type(), "eta", [&]{
        EtaK<scalar_t><<<blk, thd>>>(
            pz_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
             e_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
           eta_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    }); 
    return eta_; 
}






torch::Tensor transform_::Pt(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size(0); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    torch::Tensor pt_ = torch::zeros({dx, 1}, MakeOp(pmc)); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "pt", [&]{
        PtK<scalar_t><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               pt_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx
        );
    }); 
    return pt_; 
}

torch::Tensor transform_::Eta(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size(0); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    torch::Tensor eta = torch::zeros({dx, 1}, MakeOp(pmc)); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "eta", [&]{
        EtaK<scalar_t><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               eta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx
        );
    }); 
    return eta; 
}

torch::Tensor transform_::Phi(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size(0); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    torch::Tensor phi = torch::zeros({dx, 1}, MakeOp(pmc)); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "phi", [&]{
        PtK<scalar_t><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx
        );
    }); 
    return phi; 
}




torch::Tensor transform_::Px(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size(0); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    torch::Tensor px = torch::zeros({dx, 1}, MakeOp(pmc)); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "px", [&]{
        PxK<scalar_t><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               px.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx
        );
    }); 
    return px; 
}

torch::Tensor transform_::Py(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size(0); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    torch::Tensor py = torch::zeros({dx, 1}, MakeOp(pmc)); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "py", [&]{
        PyK<scalar_t><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               py.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx
        );
    }); 
    return py; 
}

torch::Tensor transform_::Pz(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size(0); 
    const unsigned int thx = (dx >= phys_th) ? phys_th : dx;
    const dim3 thd = dim3(thx); 
    const dim3 blk = blk_(dx, thx);

    torch::Tensor pz = torch::zeros({dx, 1}, MakeOp(pmc)); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "pz", [&]{
        PzK<scalar_t><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                pz.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                dx
        );
    }); 
    return pz; 
}


torch::Tensor transform_::PtEtaPhiE(torch::Tensor* pmc){
    const unsigned int dx = pmc -> size({0}); 
    const unsigned int dy = pmc -> size({-1});
    torch::Tensor pmu = torch::zeros({dx, 4}, MakeOp(pmc)); 

    const unsigned int thx = (dx >= phys_th) ? phys_th : dx; 
    const dim3 thd = dim3(thx, 4); 
    const dim3 blk = blk_(dx, thx, 4, 4);

    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "ptetaphie", [&]{
        PtEtaPhiEK<scalar_t, phys_th><<<blk, thd>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               pmu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dx, dy
        );
    }); 
    return pmu;   
}


torch::Tensor transform_::PxPyPzE(torch::Tensor* pmu){
    const unsigned int dx = pmu -> size({0}); 
    const unsigned int dy = pmu -> size({-1});
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

torch::Tensor transform_::PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmc = format({*px, *py, *pz, *e}); 
    return transform_::PtEtaPhiE(&pmc);   
}

torch::Tensor transform_::PxPyPzE(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* e){
    torch::Tensor pmu = format({*pt, *eta, *phi, *e}); 
    return transform_::PxPyPzE(&pmu); 
}


