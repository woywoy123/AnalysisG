#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void ToCartesian_x_Kernel(
    scalar_t* __restrict__ PX, 
    const scalar_t* __restrict__ phi, 
    const scalar_t* __restrict__ pt, 
    size_t len_v)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x; 
  if (index < len_v){ PX[index] = pt[index]*cos(phi[index]); }
}

template <typename scalar_t>
__global__ void ToCartesian_y_Kernel(
    scalar_t* __restrict__ PY, 
    const scalar_t* __restrict__ phi, 
    const scalar_t* __restrict__ pt, 
    size_t len_v)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x; 
  if (index < len_v){ PY[index] = pt[index]*sin(phi[index]); }
}

template <typename scalar_t>
__global__ void ToCartesian_z_Kernel(
    scalar_t* __restrict__ PZ, 
    const scalar_t* __restrict__ eta, 
    const scalar_t* __restrict__ e, 
    size_t len_v)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x; 
  if (index < len_v){ PZ[index] = e[index]*tanh(eta[index]); }
}

template <typename scalar_t>
__global__ void ToCartesian_e_Kernel(
    scalar_t* __restrict__ E, 
    const scalar_t* __restrict__ e, 
    size_t len_v)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x; 
  if (index < len_v){ E[index] = e[index]; }
}

std::vector<torch::Tensor> ToCartesianCUDA(torch::Tensor eta, torch::Tensor phi, torch::Tensor pt, torch::Tensor e)
{
  const int l = eta.size(0);
  const int threads = 1024; 
  torch::Tensor PX = torch::zeros_like(pt); 
  torch::Tensor PY = torch::zeros_like(pt); 
  torch::Tensor PZ = torch::zeros_like(pt); 
  torch::Tensor E = torch::zeros_like(pt); 
  
  const dim3 blocks((l + threads -1) / threads, 1); 

  AT_DISPATCH_FLOATING_TYPES(eta.type(), "CUDA_x_ToCartesian", ([&] 
  {
    ToCartesian_x_Kernel<scalar_t><<<blocks, threads>>>(
      PX.data<scalar_t>(), 
      phi.data<scalar_t>(), 
      pt.data<scalar_t>(), 
      l);
  }));
  AT_DISPATCH_FLOATING_TYPES(eta.type(), "CUDA_y_ToCartesian", ([&] 
  {
    ToCartesian_y_Kernel<scalar_t><<<blocks, threads>>>(
      PY.data<scalar_t>(), 
      phi.data<scalar_t>(), 
      pt.data<scalar_t>(), 
      l);
  }));
  AT_DISPATCH_FLOATING_TYPES(eta.type(), "CUDA_z_ToCartesian", ([&] 
  {
    ToCartesian_z_Kernel<scalar_t><<<blocks, threads>>>(
      PZ.data<scalar_t>(), 
      eta.data<scalar_t>(), 
      e.data<scalar_t>(), 
      l);
  }));
  AT_DISPATCH_FLOATING_TYPES(eta.type(), "CUDA_e_ToCartesian", ([&] 
  {
    ToCartesian_e_Kernel<scalar_t><<<blocks, threads>>>(
      E.data<scalar_t>(), 
      e.data<scalar_t>(), 
      l);
  }));
  return {PX, PY, PZ, E};
}


template <typename scalar_t> 
__device__ __forceinline__ scalar_t Mass(scalar_t px, scalar_t py, scalar_t pz, scalar_t energy){
  return sqrt(energy*energy - px*px - py*py - pz*pz) / 1000; 
}

template <typename scalar_t>
__global__ void PathCombinationMass_Kernel(
    scalar_t* __restrict__ m, 
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ y, 
    const scalar_t* __restrict__ z, 
    const scalar_t* __restrict__ e, 
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> combi, 
    size_t comb_l, size_t l_v_n)
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index < comb_l)
  {
    scalar_t x_s = 0; 
    scalar_t y_s = 0;
    scalar_t z_s = 0; 
    scalar_t e_s = 0;
    for (int k(0); k < l_v_n; k++)
    { 
      x_s += x[k] * combi[index][k][0]; 
      y_s += y[k] * combi[index][k][0]; 
      z_s += z[k] * combi[index][k][0]; 
      e_s += e[k] * combi[index][k][0]; 
    }
    m[index] = Mass(x_s, y_s, z_s, e_s); 
  }
}

torch::Tensor PathMassCartesianCUDA(torch::Tensor x, torch::Tensor y, torch::Tensor z, torch::Tensor e, torch::Tensor Combination)
{
  const int l = Combination.size(0); 
  const int l_v_n = x.size(0);
  const int threads = 1024; 
  const dim3 blocks((l+threads-1)/threads, 1); 

  torch::Tensor m = torch::zeros(l, torch::TensorOptions().device(x.device().type())); 
  AT_DISPATCH_FLOATING_TYPES(x.type(), "CUDA_Mass", ([&] 
  {
    PathCombinationMass_Kernel<scalar_t><<<blocks, threads>>>(
        m.data<scalar_t>(), 
        x.data<scalar_t>(), 
        y.data<scalar_t>(), 
        z.data<scalar_t>(), 
        e.data<scalar_t>(), 
        Combination.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
        l, l_v_n);
    }));
  return m;  
}

