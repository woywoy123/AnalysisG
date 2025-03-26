#include <pyc/cupyc.h>

torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor>* inpt){
    torch::Dict<std::string, torch::Tensor> out;  
    std::map<std::string, torch::Tensor>::iterator itr = inpt -> begin(); 
    for (; itr != inpt -> end(); ++itr){out.insert(itr -> first, itr -> second);}
    return out; 
}

torch::Dict<std::string, torch::Tensor> pyc::std_to_dict(std::map<std::string, torch::Tensor> inpt){
    return pyc::std_to_dict(&inpt); 
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

    m.def("nusol_base_basematrix", &pyc::nusol::BaseMatrix); 
    m.def("nusol_nu"             , &pyc::nusol::Nu); 
    m.def("nusol_nunu"           , &pyc::nusol::NuNu); 
    m.def("nusol_combinatorial"  , &pyc::nusol::combinatorial); 

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
    m.def("operators_cross"      , &pyc::operators::Cross); 

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



    m.def("graph_edge_aggregation"  , &pyc::graph::edge_aggregation); 
    m.def("graph_node_aggregation"  , &pyc::graph::node_aggregation); 
    m.def("graph_unique_aggregation", &pyc::graph::unique_aggregation); 
    m.def("graph_page_rank"         , &pyc::graph::page_rank); 
}
