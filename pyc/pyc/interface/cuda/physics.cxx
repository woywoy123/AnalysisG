#include <pyc/cupyc.h>
#include <utils/utils.cuh>
#include <physics/physics.cuh>
#include <transform/transform.cuh>

torch::Tensor pyc::physics::cartesian::separate::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&pz); 
    return physics_::P2(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::P2(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::P2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    changedev(&phi); 
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::P2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::P2(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::P2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&pz); 
    return physics_::P(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::P(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::P(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::P(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::P(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::P(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&px);
    return physics_::Beta2(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Beta2(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Beta2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::Beta2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Beta2(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Beta2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&px); 
    return physics_::Beta(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Beta(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Beta(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::Beta(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Beta(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Beta(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&px); 
    return physics_::M2(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::M2(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::M2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::M2(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::M2(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::M2(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    changedev(&px); 
    return physics_::M(&px, &py, &pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::M(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::polar::separate::M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    return physics_::M(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::M(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::M(&pmc); 
}


torch::Tensor pyc::physics::cartesian::separate::Mt2(torch::Tensor pz, torch::Tensor e){
    changedev(&pz); 
    return physics_::Mt2(&pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Mt2(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Mt2(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pz = transform_::Pz(&pt, &eta); 
    return physics_::Mt2(&pz, &e); 
}

torch::Tensor pyc::physics::polar::combined::Mt2(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Mt2(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Mt(torch::Tensor pz, torch::Tensor e){
    changedev(&pz); 
    return physics_::Mt(&pz, &e); 
}

torch::Tensor pyc::physics::cartesian::combined::Mt(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Mt(&pmc); 
}

torch::Tensor pyc::physics::polar::separate::Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){
    changedev(&pt); 
    torch::Tensor pz = transform_::Pz(&pt, &eta); 
    return physics_::Mt(&pz, &e); 
}

torch::Tensor pyc::physics::polar::combined::Mt(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    return physics_::Mt(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){
    changedev(&px); 
    return physics_::Theta(&px, &py, &pz); 
}

torch::Tensor pyc::physics::cartesian::combined::Theta(torch::Tensor pmc){
    changedev(&pmc); 
    return physics_::Theta(&pmc); 
}


torch::Tensor pyc::physics::polar::separate::Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){
    changedev(&pt); 
    torch::Tensor pmc = transform_::PxPyPz(&pt, &eta, &phi); 
    return physics_::Theta(&pmc); 
}

torch::Tensor pyc::physics::polar::combined::Theta(torch::Tensor pmu){
    changedev(&pmu); 
    torch::Tensor pmc = transform_::PxPyPz(&pmu); 
    return physics_::Theta(&pmc); 
}

torch::Tensor pyc::physics::cartesian::separate::DeltaR(
        torch::Tensor px1, torch::Tensor px2, 
        torch::Tensor py1, torch::Tensor py2, 
        torch::Tensor pz1, torch::Tensor pz2
){
    changedev(&px1);
    torch::Tensor pmu1 = transform_::PtEtaPhi(&px1, &py1, &pz1); 
    torch::Tensor pmu2 = transform_::PtEtaPhi(&px2, &py2, &pz2); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::physics::cartesian::combined::DeltaR(torch::Tensor pmc1, torch::Tensor pmc2){
    changedev(&pmc1); 
    torch::Tensor pmu1 = transform_::PtEtaPhi(&pmc1); 
    torch::Tensor pmu2 = transform_::PtEtaPhi(&pmc2); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}

torch::Tensor pyc::physics::polar::separate::DeltaR(
        torch::Tensor eta1, torch::Tensor eta2, 
        torch::Tensor phi1, torch::Tensor phi2
){
    changedev(&eta1); 
    return physics_::DeltaR(&eta1, &eta2, &phi1, &phi2); 
}

torch::Tensor pyc::physics::polar::combined::DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){
    changedev(&pmu1); 
    return physics_::DeltaR(&pmu1, &pmu2); 
}


