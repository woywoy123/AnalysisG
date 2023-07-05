#include <torch/extension.h>
#include <operators.h>

TORCH_LIBRARY(op_cuda, m)
{
    m.def("dot", &Operators::CUDA::Dot);     
    m.def("mul", &Operators::CUDA::Mul); 

    m.def("CosTheta", &Operators::CUDA::CosTheta);     
    m.def("SinTheta", &Operators::CUDA::SinTheta);

    m.def("Rx", &Operators::CUDA::Rx);     
    m.def("Ry", &Operators::CUDA::Ry);
    m.def("Rz", &Operators::CUDA::Rz);     

    m.def("CoFactors", &Operators::CUDA::CoFactors); 
    m.def("Determinant", &Operators::CUDA::Determinant); 
    m.def("Inverse", &Operators::CUDA::Inverse); 
} 
