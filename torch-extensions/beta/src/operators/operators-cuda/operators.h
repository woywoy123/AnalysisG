#ifndef H_OPERATORS_CUDA
#define H_OPERATORS_CUDA

torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _Mul(torch::Tensor v1, torch::Tensor v2); 

namespace Operators
{
    namespace CUDA
    {
        const torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2)
        {
            return _Dot(v1, v2); 
        }

        const torch::Tensor Mul(torch::Tensor v1, torch::Tensor v2)
        {
            return _Mul(v1, v2); 
        }
    }
}
#endif

