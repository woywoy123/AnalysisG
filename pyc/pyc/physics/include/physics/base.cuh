#ifndef CU_PHYSICS_BASE_H
#define CU_PHYSICS_BASE_H
#include <atomic/cuatomic.cuh>

template <typename scalar_t> 
__global__ void _P2K(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx, const unsigned int dy, const unsigned threads = 64
){
    __shared__ double sdata[64]; 

    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int idx = _idx/dy; 
    const unsigned int idy = _idx%dy; 
    const unsigned int scp = threads*((_idx%threads) == 0); 
    sdata[threadIdx.x] = _p2(&pmc[idx][idy]);
    pmc[idx][idy] = 0; 
    __syncthreads();
    for (size_t x(0); x < scp; ++x){pmc[(_idx + x)/dy][idy] += sdata[x];}
    if (idy){return;}
    __syncthreads(); 
    for (size_t x(1); x < dy; ++x){pmc[idx][0] += pmc[idx][x];}
}


template <typename scalar_t> 
__global__ void _PK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx, const unsigned int dy, const unsigned threads = 64
){
    __shared__ double sdata[64]; 

    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int idx = _idx/dy; 
    const unsigned int idy = _idx%dy; 
    sdata[threadIdx.x] = _p2(&pmc[idx][idy]);
    pmc[idx][idy] = 0; 
    __syncthreads();
    for (size_t x(0); x < threads*(_idx%threads == 0); ++x){
        pmc[(_idx + x)/dy][idy] += sdata[x];
    }
    if (idy){return;}
    __syncthreads(); 
    scalar_t sm = 0; 
    for (size_t x(0); x < dy; ++x){sm += pmc[idx][x]; pmc[idx][x] = 0;}
    pmc[idx][0] = _sqrt(&sm); 
}

template <typename scalar_t>
__global__ void _Beta2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx
){
    __shared__ double sdata[64][4];
   
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = (blockIdx.y * blockDim.y + threadIdx.y); 
    const bool _idz = (_idx*4 + _idy)%4 == 0; 

    if ((_idx >= dx) + (_idy >= 4)){return;}
    sdata[threadIdx.x][threadIdx.y] = pmc[_idx][_idy]; 
    double* data = &sdata[threadIdx.x][threadIdx.y]; 
    sdata[threadIdx.x][threadIdx.y] = _p2(data);
    if (threadIdx.y == 3){sdata[threadIdx.x][threadIdx.y] = _div(data);}
    pmc[_idx][_idy] = 0; 
    __syncthreads(); 

    for (size_t x(1); x < 3*_idz; ++x){
        sdata[threadIdx.x][0] += sdata[threadIdx.x][x];
        if (x < 2){continue;}
        pmc[_idx][0] = sdata[threadIdx.x][0]*sdata[threadIdx.x][3]; 
    }
}

template <typename scalar_t>
__global__ void _Beta(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx
){
    __shared__ double sdata[64][4];
   
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = (blockIdx.y * blockDim.y + threadIdx.y); 
    const bool _idz = (_idx*4 + _idy)%4 == 0; 

    if ((_idx >= dx) + (_idy >= 4)){return;}
    sdata[threadIdx.x][threadIdx.y] = pmc[_idx][_idy]; 
    double* data = &sdata[threadIdx.x][threadIdx.y]; 
    if (threadIdx.y == 3){sdata[threadIdx.x][threadIdx.y] = _div(data);}
    else {sdata[threadIdx.x][threadIdx.y] = _p2(data);}
    pmc[_idx][_idy] = 0; 
    __syncthreads(); 

    for (size_t x(1); x < 3*_idz; ++x){
        sdata[threadIdx.x][0] += sdata[threadIdx.x][x];
        if (x < 2){continue;}
        pmc[_idx][0] = _sqrt(&sdata[threadIdx.x][0]) * sdata[threadIdx.x][3]; 
    }
}


template <typename scalar_t>
__global__ void _M2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx
){
    __shared__ double sdata[64][4];
   
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = (blockIdx.y * blockDim.y + threadIdx.y); 
    const bool _idz = (_idx*4 + _idy)%4 == 0; 

    if ((_idx >= dx) + (_idy >= 4)){return;}
    sdata[threadIdx.x][threadIdx.y] = pmc[_idx][_idy]; 
    sdata[threadIdx.x][threadIdx.y] = _p2(&sdata[threadIdx.x][threadIdx.y]);
    pmc[_idx][_idy] = 0; 
    if (!_idz){return;}
    __syncthreads(); 

    for (size_t x(1); x < 3*_idz; ++x){
        sdata[threadIdx.x][0] += sdata[threadIdx.x][x];
        if (x < 2){continue;}
        pmc[_idx][0] = sdata[threadIdx.x][3] - sdata[threadIdx.x][0]; 
    }
}

template <typename scalar_t>
__global__ void _M(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx
){
    __shared__ double sdata[64][4];
   
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = (blockIdx.y * blockDim.y + threadIdx.y); 
    const bool _idz = (_idx*4 + _idy)%4 == 0; 

    if ((_idx >= dx) + (_idy >= 4)){return;}
    sdata[threadIdx.x][threadIdx.y] = pmc[_idx][_idy]; 
    sdata[threadIdx.x][threadIdx.y] = _p2(&sdata[threadIdx.x][threadIdx.y]);
    pmc[_idx][_idy] = 0; 
    if (!_idz){return;}
    __syncthreads(); 

    for (size_t x(1); x < 3*_idz; ++x){
        sdata[threadIdx.x][0] += sdata[threadIdx.x][x];
        if (x < 2){continue;}
        double data = sdata[threadIdx.x][3] - sdata[threadIdx.x][0]; 
        pmc[_idx][0] = _sqrt(&data); 
    }
}

template <typename scalar_t>
__global__ void _Mt2(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx
){
    __shared__ double sdata[128];
   
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = threadIdx.x*2 + threadIdx.y; 
    if (_idx >= dx){return;}

    sdata[_idy] = pmc[_idx][threadIdx.y]; 
    sdata[_idy] = _p2(&sdata[_idy]);
    if (_idy % 2){return;}
    __syncthreads(); 
    pmc[_idx][0] = sdata[_idy+1] - sdata[_idy];  
}

template <typename scalar_t>
__global__ void _Mt(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx
){
    __shared__ double sdata[128];
   
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = threadIdx.x * 2 + threadIdx.y; 
    if (_idx >= dx){return;}

    sdata[_idy] = pmc[_idx][threadIdx.y]; 
    sdata[_idy] = _p2(&sdata[_idy]);
    if (_idy % 2){return;}
    __syncthreads(); 
    double data = sdata[_idy+1] - sdata[_idy];
    pmc[_idx][0] = _sqrt(&data); 
}


template <typename scalar_t> 
__global__ void _theta(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx
){
    __shared__ double sdata[64][4]; 

    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = (blockIdx.y * blockDim.y + threadIdx.y); 
    const bool _idz = (threadIdx.x * 3 + threadIdx.y) % 3 == 0; 

    if (_idx >= dx){return;}
    sdata[threadIdx.x][threadIdx.y] = pmc[_idx][_idy];

    if (threadIdx.y == 2){sdata[threadIdx.x][3] = sdata[threadIdx.x][threadIdx.y];}
    sdata[threadIdx.x][threadIdx.y] = _p2(&sdata[threadIdx.x][threadIdx.y]);
    if (!_idz){return;}
    __syncthreads();  
  
    double sm = 0;  
    for (size_t x(0); x < 3; ++x){sm += sdata[threadIdx.x][x];}
    pmc[_idx][0] = _arccos(&sm, &sdata[threadIdx.x][3]);
}


template <typename scalar_t> 
__global__ void _deltar(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu1, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu2, 
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> dr, 
        const unsigned int dx, const unsigned offset = 1
){
    __shared__ double sdata[64][2]; 
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = (blockIdx.y * blockDim.y + threadIdx.y) + offset; 
    const bool _idz = (threadIdx.x * 2 + threadIdx.y) % 2 == 0; 
    scalar_t data = pmu1[_idx][_idy] - pmu2[_idx][_idy];
    if (threadIdx.y){data = minus_mod(&data);}
    sdata[threadIdx.x][threadIdx.y] = _p2(&data);
    if (!_idz){return;}
    __syncthreads(); 
    double drx = sdata[threadIdx.x][0] + sdata[threadIdx.x][1];
    dr[_idx][0] = _sqrt(&drx); 
}

#endif
