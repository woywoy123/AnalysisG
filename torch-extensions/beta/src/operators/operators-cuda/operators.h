#ifndef H_OPERATORS_CUDA
#define H_OPERATORS_CUDA

torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _Mul(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _SinTheta(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _Rot(torch::Tensor angle, const unsigned int dim); 
torch::Tensor _CoFactors(torch::Tensor matrix); 
torch::Tensor _Det(torch::Tensor matrix); 
torch::Tensor _Inv(torch::Tensor matrix); 

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

        const torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2)
        {
            return _CosTheta(v1, v2); 
        }
        
        const torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2)
        {
            return _SinTheta(v1, v2); 
        }

        const torch::Tensor Rx(torch::Tensor angle)
        {
            return _Rot(angle, 0); 
        }

        const torch::Tensor Ry(torch::Tensor angle)
        {
            return _Rot(angle, 1); 
        }

        const torch::Tensor Rz(torch::Tensor angle)
        {
            return _Rot(angle, 2); 
        }

        const torch::Tensor CoFactors(torch::Tensor matrix)
        {
            return _CoFactors(matrix); 
        }

        const torch::Tensor Determinant(torch::Tensor matrix)
        {
            return _Det(matrix); 
        }

        const torch::Tensor Inverse(torch::Tensor matrix)
        {
            return _Inv(matrix); 
        }




    }
}
#endif

