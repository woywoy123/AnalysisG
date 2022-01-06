#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t> 
__device__ __forceinline__ scalar_t pt_cos(scalar_t phi, scalar_t pt, scalar_t Combi){
  return pt*cos(phi)*Combi; 
}

template <typename scalar_t> 
__device__ __forceinline__ scalar_t pt_sin(scalar_t phi, scalar_t pt, scalar_t Combi){
  return pt*sin(phi)*Combi; 
}

template <typename scalar_t> 
__device__ __forceinline__ scalar_t e_tanh(scalar_t eta, scalar_t energy, scalar_t Combi){
  return energy*tanh(eta)*Combi; 
}

template <typename scalar_t> 
__device__ __forceinline__ scalar_t Mass(scalar_t px, scalar_t py, scalar_t pz, scalar_t energy){
  return sqrt(energy*energy - px*px - py*py - pz*pz); 
}

template <typename scalar_t> 
__device__ __forceinline__ void Sum(scalar_t *val, scalar_t inc){
  *val = *val + inc; 
}


template<typename scalar_t>
__global__ void FastMassMulti_Kernel(
    scalar_t* __restrict__ PX, 
    scalar_t* __restrict__ PY, 
    scalar_t* __restrict__ PZ, 
    scalar_t* __restrict__ E,
    const scalar_t* __restrict__ e, 
    const scalar_t* __restrict__ eta, 
    const scalar_t* __restrict__ phi,  
    const scalar_t* __restrict__ pt,  
    const scalar_t* __restrict__ Combi, 
    size_t four_vec_len){
      const int column = blockIdx.x * blockDim.x + threadIdx.x;
      const int index = blockIdx.y * four_vec_len + column; 
      if (column < four_vec_len){
        PX[index] = pt_cos(phi[column], pt[column], Combi[column]); 
        PZ[index] = e_tanh(eta[column], e[column], Combi[column]);
        PY[index] = pt_sin(phi[column], pt[column], Combi[column]); 
        E[index] = Combi[column]*e[column];
      }
    }

std::vector<torch::Tensor> FastMassMultiplicationCUDA(torch::Tensor eta, torch::Tensor phi, torch::Tensor pt, torch::Tensor e, torch::Tensor Combi)
{
  const int l = eta.size(0);
  const int v = 1; 
  const int threads = 1024; 
  torch::Tensor PX = torch::zeros_like(pt); 
  torch::Tensor PY = torch::zeros_like(pt); 
  torch::Tensor PZ = torch::zeros_like(pt); 
  torch::Tensor E = torch::zeros_like(pt); 
  
  const dim3 blocks((l + threads -1) / threads, v); 

  AT_DISPATCH_FLOATING_TYPES(eta.type(), "CUDAFastMassMultiplication", ([&] {FastMassMulti_Kernel<scalar_t><<<blocks, threads>>>(
          PX.data<scalar_t>(),
          PY.data<scalar_t>(), 
          PZ.data<scalar_t>(),
          E.data<scalar_t>(),
          e.data<scalar_t>(),
          eta.data<scalar_t>(),
          phi.data<scalar_t>(),
          pt.data<scalar_t>(),
          Combi.data<scalar_t>(),
          l); 
  })); 
  return {PX, PY, PZ, E};
}

