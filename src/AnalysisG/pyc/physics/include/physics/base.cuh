#ifndef CU_PHYSICS_BASE_H
#define CU_PHYSICS_BASE_H
#include <utils/atomic.cuh>

template <typename scalar_t, size_t size_x> 
__global__ void _P2K(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p2,
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}

    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.x][threadIdx.y] = pn*pn; 
    if (threadIdx.y){return;}
    __syncthreads(); 
    p2[_idx][0] = _sum(sdata[threadIdx.x], 3); 
}


template <typename scalar_t, size_t size_x> 
__global__ void _PK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> p,
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}
    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.x][threadIdx.y] = pn*pn; 
    if (threadIdx.y){return;}
    __syncthreads(); 
    p[_idx][0] = _sqrt(_sum(sdata[threadIdx.x], 3)); 
}

template <typename scalar_t, size_t size_x>
__global__ void _Beta2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2,
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}
    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.x][threadIdx.y] = (threadIdx.y == 3) ? _div(pn*pn) : pn*pn; 
    if (threadIdx.y){return;}
    __syncthreads();
    b2[_idx][0] = _sum(sdata[threadIdx.x], 3)*sdata[threadIdx.x][3];
}

template <typename scalar_t, size_t size_x>
__global__ void _Beta(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b, 
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}
    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.x][threadIdx.y] = (threadIdx.y == 3) ? _div(pn*pn) : pn*pn; 
    if (threadIdx.y){return;}
    __syncthreads();
    b[_idx][0] = _sqrt(_sum(sdata[threadIdx.x], 3))*sdata[threadIdx.x][3];
}


template <typename scalar_t, size_t size_x>
__global__ void _M2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2,
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}

    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.x][threadIdx.y] = (pn*pn) * ((threadIdx.y == 3)*2 - 1); 
    if (threadIdx.y){return;}
    __syncthreads();
    m2[_idx][0] = _sum(sdata[threadIdx.x], 4);
}


template <typename scalar_t, size_t size_x>
__global__ void _M(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m,
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}

    double pn = pmc[_idx][threadIdx.y]; 
    sdata[threadIdx.x][threadIdx.y] = (pn*pn) * ((threadIdx.y == 3)*2 - 1); 
    if (threadIdx.y){return;}
    __syncthreads();
    m[_idx][0] = _sqrt(_sum(sdata[threadIdx.x], 4));
}

template <typename scalar_t, size_t size_x>
__global__ void _Mt2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mt2,
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}
    double pn = pmc[_idx][threadIdx.y];
    sdata[threadIdx.x][threadIdx.y] = (pn*pn) * ((threadIdx.y == 3)*2 - 1); 
    if (threadIdx.y){return;}

    __syncthreads();
    mt2[_idx][0] = sdata[threadIdx.x][3] + sdata[threadIdx.x][2];
}

template <typename scalar_t, size_t size_x>
__global__ void _Mt(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mt, 
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}
    double pn = pmc[_idx][threadIdx.y];
    sdata[threadIdx.x][threadIdx.y] = (pn*pn) * ((threadIdx.y == 3)*2 - 1); 
    if (threadIdx.y){return;}
    __syncthreads();
    mt[_idx][0] = _sqrt(sdata[threadIdx.x][3] + sdata[threadIdx.x][2]);
}


template <typename scalar_t, size_t size_x> 
__global__ void _theta(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> theta, 
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}
    const bool sw = threadIdx.y < 3; 
    double pn = pmc[_idx][threadIdx.y-!sw];

    sdata[threadIdx.x][threadIdx.y] = (sw) ? pn*pn : pn; 
    if (threadIdx.y){return;}
    __syncthreads();

    double sx = _sum(sdata[threadIdx.x], 3); 
    theta[_idx][0] = _arccos(&sx, &sdata[threadIdx.x][3]);  
}

template <typename scalar_t, size_t size_x> 
__global__ void _deltar(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu1, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu2, 
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> dr, 
        const unsigned int dx, const unsigned int dy
){
    __shared__ double sdata[size_x][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    sdata[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}

    double del = pmu1[_idx][threadIdx.y] - pmu2[_idx][threadIdx.y];  
    del = (threadIdx.y == 2) ? minus_mod(&del) : del; 
    sdata[threadIdx.x][threadIdx.y] = del * del; 
    if (threadIdx.y){return;}
    __syncthreads(); 
    dr[_idx][0] = _sqrt(sdata[threadIdx.x][1] + sdata[threadIdx.x][2]);
}

#endif
