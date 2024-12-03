#include <pyc/tpyc.h>

TORCH_LIBRARY(physics_tensor, m){
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
}
