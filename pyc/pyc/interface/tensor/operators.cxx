#include <pyc/tpyc.h>
#include <operators/operators.h>

torch::Tensor pyc::operators::Dot(torch::Tensor v1, torch::Tensor v2){
    return operators_::Dot(&v1, &v2); 
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


torch::Tensor pyc::operators::CoFactors(torch::Tensor matrix){
    return operators_::CoFactors(&matrix); 
}

torch::Tensor pyc::operators::Determinant(torch::Tensor matrix){
    return operators_::Determinant(&matrix); 
}

torch::Tensor pyc::operators::Inverse(torch::Tensor matrix){
    return operators_::Inverse(&matrix); 
}

TORCH_LIBRARY(tpyc, m){
    m.def("operators_dot"     , &pyc::operators::Dot); 
    m.def("operators_costheta", &pyc::operators::CosTheta); 
    m.def("operators_sintheta", &pyc::operators::SinTheta); 
    m.def("operators_rx"      , &pyc::operators::Rx); 
    m.def("operators_ry"      , &pyc::operators::Ry); 
    m.def("operators_rz"      , &pyc::operators::Rz); 

    m.def("operators_cofactors"  , &pyc::operators::CoFactors); 
    m.def("operators_determinant", &pyc::operators::Determinant); 
    m.def("operators_inverse"    , &pyc::operators::Inverse); 
}
