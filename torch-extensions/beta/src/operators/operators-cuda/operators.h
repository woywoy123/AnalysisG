#ifndef H_OPERATORS_CUDA
#define H_OPERATORS_CUDA

torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _Mul(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2, signed int limit); 
torch::Tensor _SinTheta(torch::Tensor v1, torch::Tensor v2, signed int limit); 
torch::Tensor _Rot(torch::Tensor angle, const unsigned int dim); 
torch::Tensor _CoFactors(torch::Tensor matrix); 
torch::Tensor _Det(torch::Tensor matrix); 
torch::Tensor _Inv(torch::Tensor matrix); 

namespace Operators
{
    namespace CUDA
    {
        inline torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2)
        {
            return _Dot(v1, v2); 
        }

        inline torch::Tensor Mul(torch::Tensor v1, torch::Tensor v2)
        {
            return _Mul(v1, v2); 
        }

        inline torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2, signed int limit = -1)
        {
            return _CosTheta(v1, v2, limit); 
        }
        
        inline torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2, signed int limit = -1)
        {
            return _SinTheta(v1, v2, limit); 
        }

        inline torch::Tensor Rx(torch::Tensor angle)
        {
            return _Rot(angle, 0); 
        }

        inline torch::Tensor Ry(torch::Tensor angle)
        {
            return _Rot(angle, 1); 
        }

        inline torch::Tensor Rz(torch::Tensor angle)
        {
            return _Rot(angle, 2); 
        }

        inline torch::Tensor CoFactors(torch::Tensor matrix)
        {
            return _CoFactors(matrix); 
        }

        inline torch::Tensor Determinant(torch::Tensor matrix)
        {
            return _Det(matrix); 
        }

        inline torch::Tensor Inverse(torch::Tensor matrix)
        {
            return _Inv(matrix); 
        }
    }
}
#endif
