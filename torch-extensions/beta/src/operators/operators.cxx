#include <torch/extension.h>
#include <operators.h>

TORCH_LIBRARY(op_cuda, m)
{
    m.def("dot", &Operators::CUDA::Dot);     
    m.def("mul", &Operators::CUDA::Mul); 
} 
