#ifndef CU_OPERATORS_BASE_H
#define CU_OPERATORS_BASE_H
#include <atomic/cuatomic.cuh>

template <typename scalar_t>
__global__ void _dot(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> v1, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> v2, 
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> out, 
        const unsigned int dx, const unsigned int dy
){

    extern __shared__ double sdata[]; 
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = (blockIdx.y * blockDim.y + threadIdx.y); 
    const unsigned int _idz = (_idx*dy + _idy)%dy; 

    if (_idx >= dx || _idy >= dy){return;}
    sdata[_idz] = v1[_idx][_idy] * v2[_idx][_idy];  
    __syncthreads(); 
    if (_idz){return;}
    for (size_t x(1); x < dy; ++x){sdata[0] += sdata[x];}
    out[_idx] = sdata[0]; 
}

template <typename scalar_t>
__global__ void _costheta(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out,
        const unsigned int dx, const unsigned int dy, bool get_sin = false
){
    extern __shared__ double sdata[]; 
    __shared__ double smem[3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = (_idx*dy + _idy)%dy; 

    if (_idx >= dx || _idy >= dy){return;}
    sdata[_idz*2]   = x[_idx][_idy];
    sdata[_idz*2+1] = y[_idx][_idy]; 
    if (_idz >= 3){return;}
    __syncthreads(); 

    const unsigned int o1x  = (threadIdx.y) ? 1 : 0; 
    const unsigned int o2x  = (threadIdx.y > 1) ? 1 : 0; 

    double sm = 0; 
    for (size_t x(0); x < dy; ++x){sm += sdata[x*2 + o1x] * sdata[x*2 + o2x];} 
    smem[_idz] = sm; 

    if (_idz){return;}
    __syncthreads(); 
    double cs = _cmp(smem[0], smem[2], smem[1]); 
    if (!get_sin){out[_idx][0] = cs; return; }
    cs = 1 - pow(cs, 2); 
    out[_idx][0] = _sqrt(&cs); 
}

template <typename scalar_t>
__global__ void _rx(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const unsigned int dx
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
  
    scalar_t vl = angle[_idx][0];  
    out[_idx][_idy][_idz] = _rx(&vl, _idy, _idz); 
}



template <typename scalar_t>
__global__ void _ry(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const unsigned int dx
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
  
    scalar_t vl = angle[_idx][0];  
    out[_idx][_idy][_idz] = _ry(&vl, _idy, _idz); 
}


template <typename scalar_t>
__global__ void _rz(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const unsigned int dx
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
  
    scalar_t vl = angle[_idx][0];  
    out[_idx][_idy][_idz] = _rz(&vl, _idy, _idz); 
}


template <typename scalar_t>
__global__ void _rt(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> phi, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> theta, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out
){
    __shared__ double pmx[3]; 
    __shared__ double pmr[9][3];
    __shared__ double pmd[9][3];

    __shared__ double rz[3][3];
    __shared__ double ry[3][3];

    __shared__ double rxt[3][3]; 
    __shared__ double rzt[3][3];
    __shared__ double ryt[3][3];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _blk = (blockIdx.y * blockDim.y + threadIdx.y)*3 + blockIdx.z * blockDim.z + threadIdx.z; 
    const unsigned int _idy  = _blk/3;
    const unsigned int _idz  = _blk%3; 

    scalar_t phi_   = -phi[_idx]; 
    scalar_t theta_ = 0.5*M_PI - theta[_idx][0]; 
    pmx[_idz] = pmc[_idx][_idz]; 

    double rz_ = _rz(&phi_  , _idy, _idz); 
    double ry_ = _ry(&theta_, _idy, _idz); 

    rz[_idy][_idz]  = rz_; 
    ry[_idy][_idz]  = ry_;  

    rzt[_idz][_idy] = rz_; 
    ryt[_idz][_idy] = ry_;  
    __syncthreads(); 

    for (size_t x(0); x < 3; ++x){
        for (size_t y(0); y < 3; ++y){pmr[_blk][x] += pmx[y] * rz[x][y];}
    }
    for (size_t x(0); x < 3; ++x){
        for (size_t y(0); y < 3; ++y){pmd[_blk][x] += pmr[_blk][y] * ry[x][y];}
    }

    double smz = -atan2(pmd[_blk][2], pmd[_blk][1]); 
    rxt[_idz][_idy] = _rx(&smz, _idy, _idz); 
    for (size_t y(0); y < 3; ++y){pmr[_blk][y] = 0;}
    for (size_t y(0); y < 3; ++y){pmd[_blk][y] = 0;}
    __syncthreads(); 

    for (size_t x(0); x < 3; ++x){pmr[_idz][_idy] += ryt[_idz][x] * rxt[x][_idy];}
    __syncthreads(); 

    for (size_t x(0); x < 3; ++x){pmd[_idz][_idy] += rzt[_idz][x] * pmr[x][_idy];}
    out[_idx][_idz][_idy] = pmd[_idz][_idy]; 
}

#endif
