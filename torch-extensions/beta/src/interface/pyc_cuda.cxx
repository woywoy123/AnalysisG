#include <torch/extension.h>
#include "pyc_cuda.h"

// ======= Transform ========= //
torch::Tensor pyc::transform::separate::Pt(torch::Tensor px, torch::Tensor py){ return Transform::CUDA::Pt(px, py); }
torch::Tensor pyc::transform::separate::Eta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return Transform::CUDA::Eta(px, py, pz); }
torch::Tensor pyc::transform::separate::Phi(torch::Tensor px, torch::Tensor py){ return Transform::CUDA::Phi(px, py); }
torch::Tensor pyc::transform::separate::PtEtaPhi(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return Transform::CUDA::PtEtaPhi(px, py, pz); }
torch::Tensor pyc::transform::separate::PtEtaPhiE(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ return Transform::CUDA::PtEtaPhiE(px, py, pz, e); }

torch::Tensor pyc::transform::separate::Px(torch::Tensor pt, torch::Tensor phi){ return Transform::CUDA::Px(pt, phi); }
torch::Tensor pyc::transform::separate::Py(torch::Tensor pt, torch::Tensor phi){ return Transform::CUDA::Py(pt, phi); }
torch::Tensor pyc::transform::separate::Pz(torch::Tensor pt, torch::Tensor eta){ return Transform::CUDA::Pz(pt, eta); }
torch::Tensor pyc::transform::separate::PxPyPz(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){ return Transform::CUDA::PxPyPz(pt, eta, phi); }
torch::Tensor pyc::transform::separate::PxPyPzE(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ return Transform::CUDA::PxPyPzE(pt, eta, phi, e); }

torch::Tensor pyc::transform::combined::Pt(torch::Tensor pmc){ return Transform::CUDA::Pt(pmc); }
torch::Tensor pyc::transform::combined::Eta(torch::Tensor pmc){ return Transform::CUDA::Eta(pmc); }
torch::Tensor pyc::transform::combined::Phi(torch::Tensor pmc){ return Transform::CUDA::Phi(pmc); }
torch::Tensor pyc::transform::combined::PtEtaPhi(torch::Tensor pmc){ return Transform::CUDA::PtEtaPhi(pmc); }
torch::Tensor pyc::transform::combined::PtEtaPhiE(torch::Tensor pmc){ return Transform::CUDA::PtEtaPhiE(pmc); }

torch::Tensor pyc::transform::combined::Px(torch::Tensor pmu){ return Transform::CUDA::Px(pmu); }
torch::Tensor pyc::transform::combined::Py(torch::Tensor pmu){ return Transform::CUDA::Py(pmu); }
torch::Tensor pyc::transform::combined::Pz(torch::Tensor pmu){ return Transform::CUDA::Pz(pmu); }
torch::Tensor pyc::transform::combined::PxPyPz(torch::Tensor pmu){ return Transform::CUDA::PxPyPz(pmu); }
torch::Tensor pyc::transform::combined::PxPyPzE(torch::Tensor pmu){ return Transform::CUDA::PxPyPzE(pmu); }

// ======= Physics ========= //
torch::Tensor pyc::physics::cartesian::separate::P2(torch::Tensor px, torch::Tensor py, torch::Tensor pz){return Physics::CUDA::Cartesian::P2(px, py, pz);}
torch::Tensor pyc::physics::cartesian::combined::P2(torch::Tensor pmc){return Physics::CUDA::Cartesian::P2(pmc);}
torch::Tensor pyc::physics::cartesian::separate::P(torch::Tensor px, torch::Tensor py, torch::Tensor pz){return Physics::CUDA::Cartesian::P(px, py, pz);}
torch::Tensor pyc::physics::cartesian::combined::P(torch::Tensor pmc){return Physics::CUDA::Cartesian::P(pmc);}

torch::Tensor pyc::physics::polar::separate::P2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){return Physics::CUDA::Polar::P2(pt, eta, phi);}
torch::Tensor pyc::physics::polar::combined::P2(torch::Tensor pmu){return Physics::CUDA::Polar::P2(pmu);}
torch::Tensor pyc::physics::polar::separate::P(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){return Physics::CUDA::Polar::P(pt, eta, phi);}
torch::Tensor pyc::physics::polar::combined::P(torch::Tensor pmu){return Physics::CUDA::Polar::P(pmu);}

torch::Tensor pyc::physics::cartesian::separate::Beta2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){return Physics::CUDA::Cartesian::Beta2(px, py, pz, e);}
torch::Tensor pyc::physics::cartesian::combined::Beta2(torch::Tensor pmc){return Physics::CUDA::Cartesian::Beta2(pmc);}
torch::Tensor pyc::physics::cartesian::separate::Beta(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){return Physics::CUDA::Cartesian::Beta(px, py, pz, e);}
torch::Tensor pyc::physics::cartesian::combined::Beta(torch::Tensor pmc){return Physics::CUDA::Cartesian::Beta(pmc);}

torch::Tensor pyc::physics::polar::separate::Beta2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){return Physics::CUDA::Polar::Beta2(pt, eta, phi, e);}
torch::Tensor pyc::physics::polar::combined::Beta2(torch::Tensor pmu){return Physics::CUDA::Polar::Beta2(pmu);}
torch::Tensor pyc::physics::polar::separate::Beta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){return Physics::CUDA::Polar::Beta(pt, eta, phi, e);}
torch::Tensor pyc::physics::polar::combined::Beta(torch::Tensor pmu){return Physics::CUDA::Polar::Beta(pmu);}

torch::Tensor pyc::physics::cartesian::separate::M2(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ return Physics::CUDA::Cartesian::M2(px, py, pz, e); }
torch::Tensor pyc::physics::cartesian::combined::M2(torch::Tensor pmc){ return Physics::CUDA::Cartesian::M2(pmc); }
torch::Tensor pyc::physics::cartesian::separate::M(torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){ return Physics::CUDA::Cartesian::M(px, py, pz, e); }
torch::Tensor pyc::physics::cartesian::combined::M(torch::Tensor pmc){ return Physics::CUDA::Cartesian::M(pmc); }

torch::Tensor pyc::physics::cartesian::separate::Mt2(torch::Tensor pz, torch::Tensor e){ return Physics::CUDA::Cartesian::Mt2(pz, e); }
torch::Tensor pyc::physics::cartesian::combined::Mt2(torch::Tensor pmc){ return Physics::CUDA::Cartesian::Mt2(pmc); }
torch::Tensor pyc::physics::cartesian::separate::Mt(torch::Tensor pz, torch::Tensor e){ return Physics::CUDA::Cartesian::Mt(pz, e); }
torch::Tensor pyc::physics::cartesian::combined::Mt(torch::Tensor pmc){ return Physics::CUDA::Cartesian::Mt(pmc); }

torch::Tensor pyc::physics::cartesian::separate::Theta(torch::Tensor px, torch::Tensor py, torch::Tensor pz){ return Physics::CUDA::Cartesian::Theta(px, py, pz); } 
torch::Tensor pyc::physics::cartesian::combined::Theta(torch::Tensor pmc){ return Physics::CUDA::Cartesian::Theta(pmc); } 
torch::Tensor pyc::physics::cartesian::separate::DeltaR(torch::Tensor px1, torch::Tensor px2, torch::Tensor py1, torch::Tensor py2, torch::Tensor pz1, torch::Tensor pz2)
{ 
    return Physics::CUDA::Cartesian::DeltaR(px1, px2, py1, py2, pz1, pz2); 
}
torch::Tensor pyc::physics::cartesian::combined::DeltaR(torch::Tensor pmc1, torch::Tensor pmc2){ return Physics::CUDA::Cartesian::DeltaR(pmc1, pmc2); } 

torch::Tensor pyc::physics::polar::separate::M2(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ return Physics::CUDA::Polar::M2(pt, eta, phi, e); } 
torch::Tensor pyc::physics::polar::combined::M2(torch::Tensor pmu){ return Physics::CUDA::Polar::M2(pmu); } 
torch::Tensor pyc::physics::polar::separate::M(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){ return Physics::CUDA::Polar::M(pt, eta, phi, e); } 
torch::Tensor pyc::physics::polar::combined::M(torch::Tensor pmu){ return Physics::CUDA::Polar::M(pmu); } 

torch::Tensor pyc::physics::polar::separate::Mt2(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){ return Physics::CUDA::Polar::Mt2(pt, eta, e); } 
torch::Tensor pyc::physics::polar::combined::Mt2(torch::Tensor pmu){ return Physics::CUDA::Polar::Mt2(pmu); } 
torch::Tensor pyc::physics::polar::separate::Mt(torch::Tensor pt, torch::Tensor eta, torch::Tensor e){ return Physics::CUDA::Polar::Mt(pt, eta, e); } 
torch::Tensor pyc::physics::polar::combined::Mt(torch::Tensor pmu){ return Physics::CUDA::Polar::Mt(pmu); } 

torch::Tensor pyc::physics::polar::separate::Theta(torch::Tensor pt, torch::Tensor eta, torch::Tensor phi){ return Physics::CUDA::Polar::Theta(pt, eta, phi); } 
torch::Tensor pyc::physics::polar::combined::Theta(torch::Tensor pmu){ return Physics::CUDA::Polar::Theta(pmu); } 
torch::Tensor pyc::physics::polar::separate::DeltaR(torch::Tensor eta1, torch::Tensor eta2, torch::Tensor phi1, torch::Tensor phi2)
{ 
    return Physics::CUDA::Polar::DeltaR(eta1, eta2, phi1, phi2); 
} 
torch::Tensor pyc::physics::polar::combined::DeltaR(torch::Tensor pmu1, torch::Tensor pmu2){ return Physics::CUDA::Polar::DeltaR(pmu1, pmu2); } 

TORCH_LIBRARY(pyc_cuda, m)
{
    // transformation classes for CUDA
    m.def("transform_separate_Px",        &pyc::transform::separate::Px);
    m.def("transform_separate_Py",        &pyc::transform::separate::Py);
    m.def("transform_separate_Pz",        &pyc::transform::separate::Pz);
    m.def("transform_separate_PxPyPz",    &pyc::transform::separate::PxPyPz);
    m.def("transform_separate_PxPyPzE",   &pyc::transform::separate::PxPyPzE);

    m.def("transform_combined_Px",        &pyc::transform::combined::Px);
    m.def("transform_combined_Py",        &pyc::transform::combined::Py);
    m.def("transform_combined_Pz",        &pyc::transform::combined::Pz);
    m.def("transform_combined_PxPyPz",    &pyc::transform::combined::PxPyPz);
    m.def("transform_combined_PxPyPzE",   &pyc::transform::combined::PxPyPzE);

    m.def("transform_separate_Pt",        &pyc::transform::separate::Pt);
    m.def("transform_separate_Phi",       &pyc::transform::separate::Phi);
    m.def("transform_separate_Eta",       &pyc::transform::separate::Eta);
    m.def("transform_separate_PtEtaPhi",  &pyc::transform::separate::PtEtaPhi);
    m.def("transform_separate_PtEtaPhiE", &pyc::transform::separate::PtEtaPhiE);

    m.def("transform_combined_Pt",        &pyc::transform::combined::Pt);
    m.def("transform_combined_Phi",       &pyc::transform::combined::Phi);
    m.def("transform_combined_Eta",       &pyc::transform::combined::Eta);
    m.def("transform_combined_PtEtaPhi",  &pyc::transform::combined::PtEtaPhi);
    m.def("transform_combined_PtEtaPhiE", &pyc::transform::combined::PtEtaPhiE);

    m.def("physics_separate_cartesian_P2", &pyc::physics::cartesian::separate::P2); 
    m.def("physics_combined_cartesian_P2", &pyc::physics::cartesian::combined::P2); 
    m.def("physics_separate_cartesian_P",  &pyc::physics::cartesian::separate::P); 
    m.def("physics_combined_cartesian_P",  &pyc::physics::cartesian::combined::P); 

    m.def("physics_separate_polar_P2", &pyc::physics::polar::separate::P2);  
    m.def("physics_combined_polar_P2", &pyc::physics::polar::combined::P2);  
    m.def("physics_separate_polar_P",  &pyc::physics::polar::separate::P); 
    m.def("physics_combined_polar_P",  &pyc::physics::polar::combined::P); 

    m.def("physics_separate_cartesian_Beta2", &pyc::physics::cartesian::separate::Beta2); 
    m.def("physics_combined_cartesian_Beta2", &pyc::physics::cartesian::combined::Beta2); 
    m.def("physics_separate_polar_Beta2", &pyc::physics::polar::separate::Beta2);  
    m.def("physics_combined_polar_Beta2", &pyc::physics::polar::combined::Beta2);  
 
    m.def("physics_separate_cartesian_Beta",  &pyc::physics::cartesian::separate::Beta); 
    m.def("physics_combined_cartesian_Beta",  &pyc::physics::cartesian::combined::Beta); 
    m.def("physics_separate_polar_Beta",  &pyc::physics::polar::separate::Beta); 
    m.def("physics_combined_polar_Beta",  &pyc::physics::polar::combined::Beta); 

    m.def("physics_separate_cartesian_M2", &pyc::physics::cartesian::separate::M2);          
    m.def("physics_combined_cartesian_M2", &pyc::physics::cartesian::combined::M2);          
    m.def("physics_separate_polar_M2", &pyc::physics::polar::separate::M2); 
    m.def("physics_combined_polar_M2", &pyc::physics::polar::combined::M2);

    m.def("physics_separate_cartesian_M", &pyc::physics::cartesian::separate::M);          
    m.def("physics_combined_cartesian_M", &pyc::physics::cartesian::combined::M);           
    m.def("physics_separate_polar_M", &pyc::physics::polar::separate::M);
    m.def("physics_combined_polar_M", &pyc::physics::polar::combined::M);
 
    m.def("physics_separate_polar_Mt2", &pyc::physics::polar::separate::Mt2);
    m.def("physics_combined_polar_Mt2", &pyc::physics::polar::combined::Mt2);
    m.def("physics_separate_cartesian_Mt2", &pyc::physics::cartesian::separate::Mt2);         
    m.def("physics_combined_cartesian_Mt2", &pyc::physics::cartesian::combined::Mt2);         
 
    m.def("physics_separate_polar_Mt", &pyc::physics::polar::separate::Mt);
    m.def("physics_combined_polar_Mt", &pyc::physics::polar::combined::Mt);
    m.def("physics_separate_cartesian_Mt", &pyc::physics::cartesian::separate::Mt);          
    m.def("physics_combined_cartesian_Mt", &pyc::physics::cartesian::combined::Mt);          
    
    m.def("physics_separate_cartesian_Theta", &pyc::physics::cartesian::separate::Theta);       
    m.def("physics_combined_cartesian_Theta", &pyc::physics::cartesian::combined::Theta);       
    m.def("physics_separate_polar_Theta", &pyc::physics::polar::separate::Theta); 
    m.def("physics_combined_polar_Theta", &pyc::physics::polar::combined::Theta); 

    m.def("physics_separate_cartesian_DeltaR", &pyc::physics::cartesian::separate::DeltaR);      
    m.def("physics_combined_cartesian_DeltaR", &pyc::physics::cartesian::combined::DeltaR);      
    m.def("physics_separate_polar_DeltaR", &pyc::physics::polar::separate::DeltaR);
    m.def("physics_combined_polar_DeltaR", &pyc::physics::polar::combined::DeltaR);

}