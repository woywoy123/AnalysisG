#ifndef H_OPERATORS_CUDA
#define H_OPERATORS_CUDA
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <stdio.h>
#include <cuda.h>

torch::Tensor _Dot(torch::Tensor v1, torch::Tensor v2);
torch::Tensor _Mul(torch::Tensor v1, torch::Tensor v2); 
torch::Tensor _CosTheta(torch::Tensor v1, torch::Tensor v2, signed int limit);
torch::Tensor _SinTheta(torch::Tensor v1, torch::Tensor v2, signed int limit); 
torch::Tensor _Rot(torch::Tensor angle, const unsigned int dim);
torch::Tensor _CoFactors(torch::Tensor matrix);
torch::Tensor _Det(torch::Tensor matrix);
std::tuple<torch::Tensor, torch::Tensor> _Inv(torch::Tensor matrix);
torch::Tensor _Cross(torch::Tensor mat1, torch::Tensor mat2);

namespace operators {
    namespace cuda {
        torch::Tensor Dot(torch::Tensor v1, torch::Tensor v2);
        torch::Tensor Mul(torch::Tensor v1, torch::Tensor v2);
        torch::Tensor CosTheta(torch::Tensor v1, torch::Tensor v2, signed int limit = -1);
        torch::Tensor SinTheta(torch::Tensor v1, torch::Tensor v2, signed int limit = -1);
        torch::Tensor Rx(torch::Tensor angle);
        torch::Tensor Ry(torch::Tensor angle);
        torch::Tensor Rz(torch::Tensor angle);
        torch::Tensor CoFactors(torch::Tensor matrix);
        torch::Tensor Determinant(torch::Tensor matrix);
        torch::Tensor Inverse(torch::Tensor matrix);
        std::tuple<torch::Tensor, torch::Tensor> Inverse(torch::Tensor matrix, bool det);
        torch::Tensor Cross(torch::Tensor mat1, torch::Tensor mat2);
    }
}
#endif

