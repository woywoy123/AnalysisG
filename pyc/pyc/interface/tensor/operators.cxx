#include <pyc/tpyc.h>

TORCH_LIBRARY(operators_tensor, m){
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
    m.def("operators_cross"      , &pyc::operators::Cross); 
}
