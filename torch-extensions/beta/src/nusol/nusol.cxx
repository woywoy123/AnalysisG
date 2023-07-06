#include <torch/extension.h>
#include <nusol.h>

TORCH_LIBRARY(nusol_cuda, m)
{
    m.def("Inverse", &Operators::CUDA::Inverse); 
} 
