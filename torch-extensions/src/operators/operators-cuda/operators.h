#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <stdio.h>
#include <cuda.h>

#ifndef H_OPERATORS_CUDA
#define H_OPERATORS_CUDA

torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _Mul(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2, signed int limit); 
torch::Tensor _SinTheta(torch::Tensor v1, torch::Tensor v2, signed int limit); 
torch::Tensor _Rot(torch::Tensor angle, const unsigned int dim); 
torch::Tensor _CoFactors(torch::Tensor matrix); 
torch::Tensor _Det(torch::Tensor matrix); 
std::tuple<torch::Tensor, torch::Tensor> _Inv(torch::Tensor matrix);
torch::Tensor _Cross(torch::Tensor mat1, torch::Tensor mat2);

namespace Operators
{
    namespace CUDA
    {
        inline torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(v1.get_device()); 
            torch::Tensor output = _Dot(v1, v2); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor Mul(torch::Tensor v1, torch::Tensor v2)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(v1.get_device()); 
            torch::Tensor output = _Mul(v1, v2); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2, signed int limit = -1)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(v1.get_device()); 
            torch::Tensor output = _CosTheta(v1, v2, limit); 
            c10::cuda::set_device(current_device);
            return output;
        }
        
        inline torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2, signed int limit = -1)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(v1.get_device()); 
            torch::Tensor output = _SinTheta(v1, v2, limit); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor Rx(torch::Tensor angle)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(angle.get_device()); 
            torch::Tensor output = _Rot(angle, 0); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor Ry(torch::Tensor angle)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(angle.get_device()); 
            torch::Tensor output = _Rot(angle, 1); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor Rz(torch::Tensor angle)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(angle.get_device()); 
            torch::Tensor output = _Rot(angle, 2); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor CoFactors(torch::Tensor matrix)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(matrix.get_device()); 
            torch::Tensor output = _CoFactors(matrix); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor Determinant(torch::Tensor matrix)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(matrix.get_device());
            torch::Tensor output = _Det(matrix); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor Inverse(torch::Tensor matrix)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(matrix.get_device()); 
            torch::Tensor output = std::get<0>(_Inv(matrix)); 
            c10::cuda::set_device(current_device);
            return output;
        }

        inline std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor matrix, bool det)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(matrix.get_device()); 
            std::tuple<torch::Tensor, torch::Tensor> output = _Inv(matrix);  
            c10::cuda::set_device(current_device);
            return output;
        }

        inline torch::Tensor Cross(torch::Tensor mat1, torch::Tensor mat2)
        {
            const auto current_device = c10::cuda::current_device();
            c10::cuda::set_device(mat1.get_device()); 
            torch::Tensor output = _Cross(mat1, mat2); 
            c10::cuda::set_device(current_device);
            return output;
        }
    }
}
#endif

