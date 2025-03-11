#include <pyc/pyc.h>

#ifdef PYC_CUDA
#include <utils/utils.cuh>
#include <transform/transform.cuh>
#else 
#include <utils/utils.h>
#include <transform/transform.h>
#endif

torch::Tensor pyc::transform::separate::Pt(torch::Tensor px, torch::Tensor py){
    changedev(&px); 
    return transform_::Pt(&px, &py); 
}

torch::Tensor pyc::transform::combined::Pt(torch::Tensor pmc){
    changedev(&pmc); 
    torch::Tensor px = pmc.index({torch::indexing::Slice(), 0});
    torch::Tensor py = pmc.index({torch::indexing::Slice(), 1}); 
    return transform_::Pt(&px, &py); 
}

torch::Tensor pyc::transform::separate::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&px); 
    return transform_::Eta(&px, &py, &pz); 
}

torch::Tensor pyc::transform::combined::Eta(torch::Tensor pmc){
    changedev(&pmc);
    return transform_::Eta(&pmc); 
}

torch::Tensor pyc::transform::separate::Phi(torch::Tensor px, torch::Tensor py){
    changedev(&px); 
    return transform_::Phi(&px, &py); 
}

torch::Tensor pyc::transform::combined::Phi(torch::Tensor pmc){
    torch::Tensor px = pmc.index({torch::indexing::Slice(), 0});
    torch::Tensor py = pmc.index({torch::indexing::Slice(), 1}); 
    changedev(&px); 
    return transform_::Phi(&px, &py); 
}

torch::Tensor pyc::transform::separate::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&px); 
    return transform_::PtEtaPhi(&px, &py, &pz); 
}

torch::Tensor pyc::transform::combined::PtEtaPhi(torch::Tensor pmc){
    changedev(&pmc); 
    return transform_::PtEtaPhi(&pmc); 
}

torch::Tensor pyc::transform::separate::PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&e); 
    return transform_::PtEtaPhiE(&px, &py, &pz, &e); 
}

torch::Tensor pyc::transform::combined::PtEtaPhiE(torch::Tensor pmc){
    changedev(&pmc); 
    return transform_::PtEtaPhiE(&pmc); 
}

torch::Tensor pyc::transform::separate::Px(torch::Tensor pt, torch::Tensor phi){
    changedev(&pt); 
    return transform_::Px(&pt, &phi); 
}

torch::Tensor pyc::transform::combined::Px(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pt = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor phi = pmu.index({torch::indexing::Slice(), 2}); 
    return transform_::Px(&pt, &phi); 
}

torch::Tensor pyc::transform::separate::Py(torch::Tensor pt, torch::Tensor phi){
    changedev(&phi); 
    return transform_::Py(&pt, &phi); 
}

torch::Tensor pyc::transform::combined::Py(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pt = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor phi = pmu.index({torch::indexing::Slice(), 2}); 
    return transform_::Py(&pt, &phi); 
}

torch::Tensor pyc::transform::separate::Pz(torch::Tensor pt, torch::Tensor eta){
    changedev(&pt);
    return transform_::Pz(&pt, &eta); 
}

torch::Tensor pyc::transform::combined::Pz(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pt  = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor eta = pmu.index({torch::indexing::Slice(), 1}); 
    return transform_::Pz(&pt, &eta); 
}

torch::Tensor pyc::transform::separate::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    changedev(&phi); 
    return transform_::PxPyPz(&pt, &eta, &phi); 
}

torch::Tensor pyc::transform::combined::PxPyPz(torch::Tensor pmu){
    changedev(&pmu); 
    return transform_::PxPyPz(&pmu); 
}

torch::Tensor pyc::transform::separate::PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    return transform_::PxPyPzE(&pt, &eta, &phi, &e); 
}

torch::Tensor pyc::transform::combined::PxPyPzE(torch::Tensor pmu){
    changedev(&pmu); 
    return transform_::PxPyPzE(&pmu); 
}


