#include <transform/transform.cuh>
#include <transform/base.cuh>
#include <cutils/utils.cuh>

torch::Tensor transform_::Px(torch::Tensor* pt, torch::Tensor* phi){
    torch::Tensor pt_  = format(pt , {-1}); 
    torch::Tensor phi_ = format(phi, {-1});    
    torch::Tensor px_  = torch::zeros_like(pt_); 
    const unsigned int dx = pt_.size(0); 
    const dim3 blk = BLOCKS(dx); 
    AT_DISPATCH_FLOATING_TYPES(pt_.scalar_type(), "px", ([&]{
        PxK<scalar_t><<<blk, _threads>>>(
            pt_.packed_accessor64<scalar_t , 1, torch::RestrictPtrTraits>(), 
            phi_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            px_.packed_accessor64<scalar_t , 1, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return px_; 
}

torch::Tensor transform_::Py(torch::Tensor* pt, torch::Tensor* phi){
    torch::Tensor pt_  = format(pt , {-1}); 
    torch::Tensor phi_ = format(phi, {-1}); 
    torch::Tensor py_  = torch::zeros_like(pt_); 
    const unsigned int dx = pt_.size(0); 
    const dim3 blk = BLOCKS(dx); 
    
    AT_DISPATCH_FLOATING_TYPES(pt_.scalar_type(), "py", ([&]{
        PyK<scalar_t><<<blk, _threads>>>(
            pt_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
           phi_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            py_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 

    return py_;  
}

torch::Tensor transform_::Pz(torch::Tensor* pt, torch::Tensor* eta){
    torch::Tensor pt_  = format(pt, {-1}); 
    torch::Tensor eta_ = format(eta, {-1});
    torch::Tensor pz_  = torch::zeros_like(pt_); 
    const unsigned int dx = pt_.size(0); 
    const dim3 blk = BLOCKS(dx); 
    AT_DISPATCH_FLOATING_TYPES(pz_.scalar_type(), "pz", ([&]{
        PzK<scalar_t><<<blk, _threads>>>(
            pt_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
           eta_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            pz_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pz_;  
}

torch::Tensor transform_::PxPyPz(torch::Tensor* pt, torch::Tensor* eta, torch::Tensor* phi){
    torch::Tensor pmu = format({*pt, *eta, *phi}); 
    torch::Tensor pmc = torch::zeros_like(pmu);
    const unsigned int dx = pt -> size(0); 
    const dim3 blk = BLOCKS(dx, 3); 
    AT_DISPATCH_FLOATING_TYPES(pt -> scalar_type(), "pxpypz", ([&]{
        PxPyPzK<scalar_t><<<blk, _threads>>>(
            pmu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pmc;   
}

torch::Tensor transform_::PxPyPz(torch::Tensor* pmu){
    torch::Tensor pmu_ = pmu -> clone(); 
    torch::Tensor pmc_ = torch::zeros_like(pmu_);
    const unsigned int dx = pmu -> size(0); 
    const dim3 blk = BLOCKS(dx, 3); 
    AT_DISPATCH_FLOATING_TYPES(pmu -> scalar_type(), "pxpypz", ([&]{
        PxPyPzK<scalar_t><<<blk, _threads>>>(
            pmu_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pmc_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pmc_;   
}


torch::Tensor transform_::PxPyPzE(
    torch::Tensor*  pt, torch::Tensor* eta, torch::Tensor* phi, torch::Tensor* energy
){
    torch::Tensor pmu = format({*pt, *eta, *phi, *energy}); 
    torch::Tensor pmc = torch::zeros_like(pmu);
    const unsigned int dx = pt -> size(0); 
    const dim3 blk = BLOCKS(dx, 4); 
    AT_DISPATCH_FLOATING_TYPES(pt -> scalar_type(), "pxpypze", ([&]{
        PxPyPzEK<scalar_t><<<blk, _threads>>>(
            pmu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pmc;   
}

torch::Tensor transform_::PxPyPzE(torch::Tensor* pmu){
    torch::Tensor pmu_ = pmu -> clone(); 
    torch::Tensor pmc_ = torch::zeros_like(pmu_);
    const unsigned int dx = pmu -> size(0); 
    const dim3 blk = BLOCKS(dx, 4); 
    AT_DISPATCH_FLOATING_TYPES(pmu -> scalar_type(), "pxpypze", ([&]{
        PxPyPzEK<scalar_t><<<blk, _threads>>>(
            pmu_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pmc_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pmc_;   
}

torch::Tensor transform_::PtEtaPhi(torch::Tensor* pmc){
    torch::Tensor pmc_ = pmc -> clone(); 
    torch::Tensor pmu_ = torch::zeros_like(pmc_);
    const unsigned int dx = pmc -> size(0); 
    const dim3 blk = BLOCKS(dx, 2); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "ptetaphi", ([&]{
        PtEtaPhiK<scalar_t><<<blk, _threads>>>(
            pmc_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pmu_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pmu_;   
}

torch::Tensor transform_::PtEtaPhiE(torch::Tensor* pmc){
    torch::Tensor pmu_ = torch::zeros_like(*pmc);
    const unsigned int dx = pmc -> size(0); 
    const dim3 blk = BLOCKS(dx, 3); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "ptetaphie", ([&]{
        PtEtaPhiEK<scalar_t><<<blk, _threads>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pmu_.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pmu_;   
}

torch::Tensor transform_::Pt(torch::Tensor* px, torch::Tensor* py){
    torch::Tensor px_ = format(px , {-1}); 
    torch::Tensor py_ = format(py, {-1});    
    torch::Tensor pt_ = torch::zeros_like(py_); 
    const unsigned int dx = pt_.size(0); 
    const dim3 blk = BLOCKS(dx); 
    AT_DISPATCH_FLOATING_TYPES(pt_.scalar_type(), "pt", ([&]{
        PtK<scalar_t><<<blk, _threads>>>(
            px_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            py_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            pt_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pt_; 
}

torch::Tensor transform_::Phi(torch::Tensor* px, torch::Tensor* py){
    torch::Tensor px_  = format(px, {-1}); 
    torch::Tensor py_  = format(py, {-1});    
    torch::Tensor phi_ = torch::zeros_like(py_); 
    const unsigned int dx = phi_.size(0); 
    const dim3 blk = BLOCKS(dx); 
    AT_DISPATCH_FLOATING_TYPES(phi_.scalar_type(), "phi", ([&]{
        PhiK<scalar_t><<<blk, _threads>>>(
             px_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
             py_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            phi_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return phi_; 
}

torch::Tensor transform_::Eta(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmc = format({*px, *py, *pz}); 
    torch::Tensor eta = px -> view({-1}).clone();
    const unsigned int dx = px -> size(0); 
    const dim3 blk = BLOCKS(dx); 
    AT_DISPATCH_FLOATING_TYPES(px -> scalar_type(), "eta", ([&]{
        EtaK<scalar_t><<<blk, _threads>>>(
            pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            eta.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return eta;   
}

torch::Tensor transform_::Eta(torch::Tensor* pmc){
    torch::Tensor eta = pmc -> index({torch::indexing::Slice(), 0}).view({-1}).clone();
    const unsigned int dx = pmc -> size(0); 
    const dim3 blk = BLOCKS(dx); 
    AT_DISPATCH_FLOATING_TYPES(pmc -> scalar_type(), "eta", ([&]{
        EtaK<scalar_t><<<blk, _threads>>>(
            pmc -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            eta.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return eta;   
}




torch::Tensor transform_::PtEtaPhi(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz){
    torch::Tensor pmc = format({*px, *py, *pz}); 
    torch::Tensor pmu = torch::zeros_like(pmc);
    const unsigned int dx = px -> size(0); 
    const dim3 blk = BLOCKS(dx, 2); 
    AT_DISPATCH_FLOATING_TYPES(px -> scalar_type(), "eta", ([&]{
        PtEtaPhiK<scalar_t><<<blk, _threads>>>(
            pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pmu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pmu;   
}

torch::Tensor transform_::PtEtaPhiE(torch::Tensor* px, torch::Tensor* py, torch::Tensor* pz, torch::Tensor* e){
    torch::Tensor pmc = format({*px, *py, *pz, *e}); 
    torch::Tensor pmu = torch::zeros_like(pmc);
    const unsigned int dx = px -> size(0); 
    const dim3 blk = BLOCKS(dx, 3); 
    AT_DISPATCH_FLOATING_TYPES(px -> scalar_type(), "eta", ([&]{
        PtEtaPhiEK<scalar_t><<<blk, _threads>>>(
            pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            pmu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dx
        );
    })); 
    return pmu;   
}
