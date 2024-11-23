#include <pyc/pyc.h>
#include <nusol/nusol.cuh>
#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <transform/transform.cuh>
#include <operators/operators.cuh>

torch::Tensor pyc::transform::separate::Pt(torch::Tensor px, torch::Tensor py){
    return transform_::Pt(&px, &py); 
}

torch::Tensor pyc::transform::combined::Pt(torch::Tensor pmc){
    return transform_::PtEtaPhi(&pmc).index({torch::indexing::Slice(), 0}); 
}

torch::Tensor pyc::transform::separate::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return transform_::Eta(&px, &py, &pz); 
}

torch::Tensor pyc::transform::combined::Eta(torch::Tensor pmc){
    return transform_::Eta(&pmc); 
}

torch::Tensor pyc::transform::separate::Phi(torch::Tensor px, torch::Tensor py){
    return transform_::Phi(&px, &py); 
}

torch::Tensor pyc::transform::combined::Phi(torch::Tensor pmc){
    torch::Tensor px = pmc.index({torch::indexing::Slice(), 0});
    torch::Tensor py = pmc.index({torch::indexing::Slice(), 1}); 
    return transform_::Phi(&px, &py); 
}

torch::Tensor pyc::transform::separate::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return transform_::PtEtaPhi(&px, &py, &pz); 
}

torch::Tensor pyc::transform::combined::PtEtaPhi(torch::Tensor pmc){
    return transform_::PtEtaPhi(&pmc); 
}

torch::Tensor pyc::transform::separate::PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return transform_::PtEtaPhiE(&px, &py, &pz, &e); 
}

torch::Tensor pyc::transform::combined::PtEtaPhiE(torch::Tensor pmc){
    return transform_::PtEtaPhiE(&pmc); 
}

torch::Tensor pyc::transform::separate::Px(torch::Tensor pt, torch::Tensor phi){
    return transform_::Px(&pt, &phi); 
}

torch::Tensor pyc::transform::combined::Px(torch::Tensor pmu){
    torch::Tensor pt = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor phi = pmu.index({torch::indexing::Slice(), 2}); 
    return transform_::Px(&pt, &phi); 
}

torch::Tensor pyc::transform::separate::Py(torch::Tensor pt, torch::Tensor phi){
    return transform_::Py(&pt, &phi); 
}

torch::Tensor pyc::transform::combined::Py(torch::Tensor pmu){
    torch::Tensor pt = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor phi = pmu.index({torch::indexing::Slice(), 2}); 
    return transform_::Py(&pt, &phi); 
}

torch::Tensor pyc::transform::separate::Pz(torch::Tensor pt, torch::Tensor eta){
    return transform_::Pz(&pt, &eta); 
}

torch::Tensor pyc::transform::combined::Pz(torch::Tensor pmu){
    torch::Tensor pt  = pmu.index({torch::indexing::Slice(), 0});
    torch::Tensor eta = pmu.index({torch::indexing::Slice(), 1}); 
    return transform_::Pz(&pt, &eta); 
}

torch::Tensor pyc::transform::separate::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    return transform_::PxPyPz(&pt, &eta, &phi); 
}

torch::Tensor pyc::transform::combined::PxPyPz(torch::Tensor pmu){
    return transform_::PxPyPz(&pmu); 
}

torch::Tensor pyc::transform::separate::PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    return transform_::PxPyPzE(&pt, &eta, &phi, &e); 
}

torch::Tensor pyc::transform::combined::PxPyPzE(torch::Tensor pmu){
    return transform_::PxPyPzE(&pmu); 
}

torch::Tensor pyc::physics::cartesian::separate::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return physics_::P2(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::P2(torch::Tensor pmc){
    return physics_::P2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::P2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::P2(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::P2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return physics_::P(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::P(torch::Tensor pmc){
    return physics_::P(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::P(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::P(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::P(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return physics_::Beta2(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Beta2(torch::Tensor pmc){
    return physics_::Beta2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::Beta2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Beta2(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Beta2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return physics_::Beta(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Beta(torch::Tensor pmc){
    return physics_::Beta(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::Beta(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Beta(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Beta(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return physics_::M2(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::M2(torch::Tensor pmc){
    return physics_::M2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::M2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::M2(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::M2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return physics_::M(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::M(torch::Tensor pmc){
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::polar::separate::M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::M(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::M(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Mt2(torch::Tensor pz, torch::Tensor e){
    return physics_::Mt2(&pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Mt2(torch::Tensor pmc){
    return physics_::Mt2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){
    torch::Tensor pz = transform_::Pz(&pt, &eta); 
    return physics_::Mt2(&pz, &e); 
}

torch::Tensor pyc::physics::polar::combined::Mt2(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Mt2(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Mt(torch::Tensor pz, torch::Tensor e){
    return physics_::Mt(&pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Mt(torch::Tensor pmc){
    return physics_::Mt(&pmc); 
}

torch::Tensor pyc::physics::polar::separate::Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){
    torch::Tensor pz = transform_::Pz(&pt, &eta); 
    return physics_::Mt(&pz, &e); 
}

torch::Tensor pyc::physics::polar::combined::Mt(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Mt(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    return physics_::Theta(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::Theta(torch::Tensor pmc){
    return physics_::Theta(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::Theta(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Theta(torch::Tensor pmu){
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::Theta(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::DeltaR(
        torch::Tensor px1, torch::Tensor px2, 
        torch::Tensor py1, torch::Tensor py2, 
        torch::Tensor pz1, torch::Tensor pz2
){
    torch::Tensor pmu1 = transform_::PtEtaPhi(&px1, &py1, &pz1); 
    torch::Tensor pmu2 = transform_::PtEtaPhi(&px2, &py2, &pz2); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::physics::cartesian::combined::DeltaR(torch::Tensor pmc1, torch::Tensor pmc2){
    torch::Tensor pmu1 = transform_::PtEtaPhi(&pmc1); 
    torch::Tensor pmu2 = transform_::PtEtaPhi(&pmc2); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::physics::polar::separate::DeltaR(
        torch::Tensor eta1, torch::Tensor eta2, 
        torch::Tensor phi1, torch::Tensor phi2
){
    return physics_::DeltaR(&eta1, &eta2, &phi1, &phi2); 
}

torch::Tensor pyc::physics::polar::combined::DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::operators::Dot(torch::Tensor v1, torch::Tensor v2){
    return operators_::Dot(&v1, &v2); 
}

torch::Tensor pyc::operators::Mul(torch::Tensor v1, torch::Tensor v2){
    return v1; 
}

torch::Tensor pyc::operators::CosTheta(torch::Tensor v1, torch::Tensor v2){
    return operators_::CosTheta(&v1, &v2); 
}

torch::Tensor pyc::operators::SinTheta(torch::Tensor v1, torch::Tensor v2){
    return operators_::SinTheta(&v1, &v2); 
}

torch::Tensor pyc::operators::Rx(torch::Tensor angle){
    return operators_::Rx(&angle); 
}

torch::Tensor pyc::operators::Ry(torch::Tensor angle){
    return operators_::Ry(&angle); 
}

torch::Tensor pyc::operators::Rz(torch::Tensor angle){
    return operators_::Rz(&angle); 
}


torch::Tensor pyc::operators::RT(torch::Tensor pmc_b, torch::Tensor pmc_mu){
    torch::Tensor phi = pyc::transform::combined::Phi(pmc_mu);
    torch::Tensor theta = physics_::Theta(&pmc_mu); 
    return operators_::RT(&pmc_b, &phi, &theta); 
}

torch::Tensor pyc::operators::CoFactors(torch::Tensor matrix){
    return operators_::CoFactors(&matrix); 
}

torch::Tensor pyc::operators::Determinant(torch::Tensor matrix){
    return operators_::Determinant(&matrix); 
}

std::tuple<torch::Tensor, torch::Tensor> pyc::operators::Inverse(torch::Tensor matrix){
    return operators_::Inverse(&matrix); 
}

torch::Tensor pyc::operators::Cross(torch::Tensor mat1, torch::Tensor mat2){
    return operators_::Cross(&mat1, &mat2); 
}

std::tuple<torch::Tensor, torch::Tensor> pyc::operators::Eigenvalue(torch::Tensor matrix){
    return operators_::Eigenvalue(&matrix); 
}


torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor>* inpt){
    torch::Dict<std::string, torch::Tensor> out;  
    std::map<std::string, torch::Tensor>::iterator itr = inpt -> begin(); 
    for (; itr != inpt -> end(); ++itr){out.insert(itr -> first, itr -> second);}
    return out; 
}

torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor> inpt){
    return pyc::std_to_dict(&inpt); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses){
    return pyc::std_to_dict(nusol_::BaseMatrix(&pmc_b, &pmc_mu, &masses)); 
}

torch::Dict<std::string, torch::Tensor> pyc::nusol::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy, 
        torch::Tensor masses, torch::Tensor sigma, double null
){

    std::map<std::string, torch::Tensor> out = nusol_::Nu(&pmc_b, &pmc_mu, &met_xy, &masses, &sigma, null);
    return pyc::std_to_dict(&out); 
}


std::tuple<torch::Tensor, torch::Tensor> Intersection(torch::Tensor A, torch::Tensor B, double null){
    return {}; 
}




TORCH_LIBRARY(cupyc, m){
    m.def("transform_separate_px",        &pyc::transform::separate::Px);
    m.def("transform_separate_py",        &pyc::transform::separate::Py);
    m.def("transform_separate_pz",        &pyc::transform::separate::Pz);
    m.def("transform_separate_pxpypz",    &pyc::transform::separate::PxPyPz);
    m.def("transform_separate_pxpypze",   &pyc::transform::separate::PxPyPzE);

    m.def("transform_combined_px",        &pyc::transform::combined::Px);
    m.def("transform_combined_py",        &pyc::transform::combined::Py);
    m.def("transform_combined_pz",        &pyc::transform::combined::Pz);
    m.def("transform_combined_pxpypz",    &pyc::transform::combined::PxPyPz);
    m.def("transform_combined_pxpypze",   &pyc::transform::combined::PxPyPzE);

    m.def("transform_separate_pt",        &pyc::transform::separate::Pt);
    m.def("transform_separate_phi",       &pyc::transform::separate::Phi);
    m.def("transform_separate_eta",       &pyc::transform::separate::Eta);
    m.def("transform_separate_ptetaphi",  &pyc::transform::separate::PtEtaPhi);
    m.def("transform_separate_ptetaphie", &pyc::transform::separate::PtEtaPhiE);

    m.def("transform_combined_pt",        &pyc::transform::combined::Pt);
    m.def("transform_combined_phi",       &pyc::transform::combined::Phi);
    m.def("transform_combined_eta",       &pyc::transform::combined::Eta);
    m.def("transform_combined_ptetaphi",  &pyc::transform::combined::PtEtaPhi);
    m.def("transform_combined_ptetaphie", &pyc::transform::combined::PtEtaPhiE);

    m.def("physics_cartesian_separate_p2", &pyc::physics::cartesian::separate::P2); 
    m.def("physics_cartesian_combined_p2", &pyc::physics::cartesian::combined::P2); 
    m.def("physics_polar_separate_p2",     &pyc::physics::polar::separate::P2);  
    m.def("physics_polar_combined_p2",     &pyc::physics::polar::combined::P2);  

    m.def("physics_cartesian_separate_p", &pyc::physics::cartesian::separate::P); 
    m.def("physics_cartesian_combined_p", &pyc::physics::cartesian::combined::P); 
    m.def("physics_polar_separate_p",     &pyc::physics::polar::separate::P);  
    m.def("physics_polar_combined_p",     &pyc::physics::polar::combined::P);  

    m.def("physics_cartesian_separate_beta2", &pyc::physics::cartesian::separate::Beta2); 
    m.def("physics_cartesian_combined_beta2", &pyc::physics::cartesian::combined::Beta2); 
    m.def("physics_polar_separate_beta2",     &pyc::physics::polar::separate::Beta2);  
    m.def("physics_polar_combined_beta2",     &pyc::physics::polar::combined::Beta2);  

    m.def("physics_cartesian_separate_beta", &pyc::physics::cartesian::separate::Beta); 
    m.def("physics_cartesian_combined_beta", &pyc::physics::cartesian::combined::Beta); 
    m.def("physics_polar_separate_beta",     &pyc::physics::polar::separate::Beta);  
    m.def("physics_polar_combined_beta",     &pyc::physics::polar::combined::Beta);  

    m.def("physics_cartesian_separate_m2", &pyc::physics::cartesian::separate::M2); 
    m.def("physics_cartesian_combined_m2", &pyc::physics::cartesian::combined::M2); 
    m.def("physics_polar_separate_m2",     &pyc::physics::polar::separate::M2);  
    m.def("physics_polar_combined_m2",     &pyc::physics::polar::combined::M2);  

    m.def("physics_cartesian_separate_m", &pyc::physics::cartesian::separate::M); 
    m.def("physics_cartesian_combined_m", &pyc::physics::cartesian::combined::M); 
    m.def("physics_polar_separate_m",     &pyc::physics::polar::separate::M);  
    m.def("physics_polar_combined_m",     &pyc::physics::polar::combined::M);  

    m.def("physics_cartesian_separate_mt2", &pyc::physics::cartesian::separate::Mt2); 
    m.def("physics_cartesian_combined_mt2", &pyc::physics::cartesian::combined::Mt2); 
    m.def("physics_polar_separate_mt2",     &pyc::physics::polar::separate::Mt2);  
    m.def("physics_polar_combined_mt2",     &pyc::physics::polar::combined::Mt2);  

    m.def("physics_cartesian_separate_mt", &pyc::physics::cartesian::separate::Mt); 
    m.def("physics_cartesian_combined_mt", &pyc::physics::cartesian::combined::Mt); 
    m.def("physics_polar_separate_mt",     &pyc::physics::polar::separate::Mt);  
    m.def("physics_polar_combined_mt",     &pyc::physics::polar::combined::Mt);  

    m.def("physics_cartesian_separate_theta", &pyc::physics::cartesian::separate::Theta); 
    m.def("physics_cartesian_combined_theta", &pyc::physics::cartesian::combined::Theta); 
    m.def("physics_polar_separate_theta",     &pyc::physics::polar::separate::Theta);  
    m.def("physics_polar_combined_theta",     &pyc::physics::polar::combined::Theta);  

    m.def("physics_cartesian_separate_deltaR", &pyc::physics::cartesian::separate::DeltaR); 
    m.def("physics_cartesian_combined_deltaR", &pyc::physics::cartesian::combined::DeltaR); 
    m.def("physics_polar_separate_deltaR",     &pyc::physics::polar::separate::DeltaR);  
    m.def("physics_polar_combined_deltaR",     &pyc::physics::polar::combined::DeltaR);  

    m.def("operators_dot"     , &pyc::operators::Dot); 
    m.def("operators_costheta", &pyc::operators::CosTheta); 
    m.def("operators_sintheta", &pyc::operators::SinTheta); 
    m.def("operators_rx"      , &pyc::operators::Rx); 
    m.def("operators_ry"      , &pyc::operators::Ry); 
    m.def("operators_rz"      , &pyc::operators::Rz); 
    m.def("operators_rt"      , &pyc::operators::RT); 

    m.def("operators_cofactors"  , &pyc::operators::CoFactors); 
    m.def("operators_determinant", &pyc::operators::Determinant); 
    m.def("operators_inverse"    , &pyc::operators::Inverse); 
    m.def("operators_eigenvalue" , &pyc::operators::Eigenvalue); 

    m.def("nusol_base_basematrix", &pyc::nusol::BaseMatrix); 
    m.def("nusol_nu"             , &pyc::nusol::Nu); 





}
