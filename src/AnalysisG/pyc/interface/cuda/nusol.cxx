#include <pyc/cupyc.h>

TORCH_LIBRARY(nusol_cuda, m){
    m.def("nusol_base_basematrix", &pyc::nusol::BaseMatrix); 
    m.def("nusol_nu"             , &pyc::nusol::Nu); 
    m.def("nusol_nunu"           , &pyc::nusol::NuNu); 
    m.def("nusol_combinatorial"  , &pyc::nusol::combinatorial); 
}
