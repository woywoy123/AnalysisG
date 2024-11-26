#ifndef CU_PHYSICS_BASE_H
#define CU_PHYSICS_BASE_H
#include <atomic/cuatomic.cuh>

template <typename scalar_t> 
__global__ void _P2K(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p2,
        const unsigned int dy
){
    extern __shared__ double sdata[]; 
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 

    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.y] = pn*pn; 
    if (threadIdx.y){return;}
    __syncthreads(); 

    p2[_idx][0] = _sum(sdata, dy); 
}


template <typename scalar_t> 
__global__ void _PK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p,
        const unsigned int dy
){
    extern __shared__ double sdata[]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 

    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.y] = pn*pn; 
    if (threadIdx.y){return;}
    __syncthreads(); 

    p[_idx][0] = _sqrt(_sum(sdata, dy)); 
}

template <typename scalar_t>
__global__ void _Beta2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2
){
    extern __shared__ double sdata[];
   
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 

    double pn = pmc[_idx][threadIdx.y]; 
    pn = pn*pn;

    sdata[threadIdx.y] = (threadIdx.y == 3) ? _div(pn) : pn; 
    if (threadIdx.y){return;}
    __syncthreads();

    b2[_idx][0] = _sum(sdata, 3)*sdata[3];
}

template <typename scalar_t>
__global__ void _Beta(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b
){
    extern __shared__ double sdata[];
   
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.y] = (threadIdx.y == 3) ? _div(pn) : pn*pn; 
    if (threadIdx.y){return;}
    __syncthreads();
    b[_idx][0] = _sqrt(_sum(sdata, 3))*sdata[3];
}


template <typename scalar_t>
__global__ void _M2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2
){
    extern __shared__ double sdata[];
   
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    double pn = pmc[_idx][threadIdx.y] ; 
    sdata[threadIdx.y] = (pn*pn) * ((threadIdx.y == 3)*2 - 1); 
    if (threadIdx.y){return;}
    __syncthreads();
    m2[_idx][0] = _sum(sdata, 4);
}


template <typename scalar_t>
__global__ void _M(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m
){
    extern __shared__ double sdata[];
   
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    double pn = pmc[_idx][threadIdx.y] ; 
    sdata[threadIdx.y] = (pn*pn) * ((threadIdx.y == 3)*2 - 1); 
    if (threadIdx.y){return;}
    __syncthreads();
    m[_idx][0] = _sqrt(_sum(sdata, 4));
}

template <typename scalar_t>
__global__ void _Mt2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mt2
){
    extern __shared__ double sdata[];
   
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    double pn = pmc[_idx][threadIdx.y];
    sdata[threadIdx.y] = (pn*pn) * ((threadIdx.y == 3)*2 - 1); 
    if (threadIdx.y){return;}
    __syncthreads();
    mt2[_idx][0] = sdata[3] + sdata[2];
}

template <typename scalar_t>
__global__ void _Mt(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mt
){
    extern __shared__ double sdata[];
   
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    double pn = pmc[_idx][threadIdx.y];
    sdata[threadIdx.y] = (pn*pn) * ((threadIdx.y == 3)*2 - 1); 
    if (threadIdx.y){return;}
    __syncthreads();
    mt[_idx][0] = _sqrt(sdata[3] + sdata[2]);
}


template <typename scalar_t> 
__global__ void _theta(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> theta
){
    extern __shared__ double sdata[];
   
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const bool sw = threadIdx.y < 3; 

    double pn = 0; 
    if (sw){pn = pmc[_idx][threadIdx.y];}
    else {pn = pmc[_idx][threadIdx.y-1];} 
    sdata[threadIdx.y] = (sw) ? pn*pn : pn; 
    if (threadIdx.y){return;}
    __syncthreads();

    double sx = _sum(sdata, 3); 
    theta[_idx][0] = _arccos(&sx, &sdata[3]);  
}

template <typename scalar_t> 
__global__ void _deltar(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu1, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu2, 
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> dr
){
    extern __shared__ double sdata[];
   
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    double del = pmu1[_idx][threadIdx.y] - pmu2[_idx][threadIdx.y];  
    del = (threadIdx.y == 2) ? minus_mod(&del) : del; 
    sdata[threadIdx.y] = del * del; 
    if (threadIdx.y){return;}
    __syncthreads(); 
    dr[_idx][0] = _sqrt(sdata[1] + sdata[2]);
}

#endif
