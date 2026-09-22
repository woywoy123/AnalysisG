#include <utils/atomic.cuh>

template <typename scalar_t>
__global__ void PxK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px,
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    px[idx][0] = px_(pt[idx][0], phi[idx][0]);
}

template <typename scalar_t>
__global__ void PyK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    py[idx][0] = py_(pt[idx][0], phi[idx][0]);
}

template <typename scalar_t>
__global__ void PzK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> energy, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> eta, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pz, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    pz[idx][0] = pz_(eta[idx][0], energy[idx][0]);
} 


template <typename scalar_t> 
__global__ void PtK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    pt[idx][0] = pt_(px[idx][0], py[idx][0]);
}

template <typename scalar_t> 
__global__ void PhiK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    phi[idx][0] = phi_(px[idx][0], py[idx][0]);
}        


template <typename scalar_t> 
__global__ void EtaK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pz, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> e, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> eta, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    eta[idx][0] = eta_(pz[idx][0], e[idx][0]);
}


template <typename scalar_t> 
__global__ void PtK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    pt[idx][0] = pt_(pmc[idx][0], pmc[idx][1]);
}

template <typename scalar_t> 
__global__ void PhiK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    phi[idx][0] = phi_(pmc[idx][0], pmc[idx][1]);
}        


template <typename scalar_t> 
__global__ void EtaK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> eta, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    eta[idx][0] = eta_(pmc[idx][2], pmc[idx][3]);
}


template <typename scalar_t>
__global__ void PxK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px,
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    px[idx][0] = px_(pmc[idx][0], pmc[idx][2]);
}

template <typename scalar_t>
__global__ void PyK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    py[idx][0] = py_(pmc[idx][0], pmc[idx][2]);
}

template <typename scalar_t>
__global__ void PzK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pz, 
    const unsigned int _dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= _dx){return;}
    pz[idx][0] = pz_(pmc[idx][1], pmc[idx][3]);
} 



template <typename scalar_t, size_t size_x>
__global__ void PxPyPzEK(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        const unsigned int dx, const unsigned int dy
){
    __shared__ double pmx[size_x][4]; 
    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    const bool blx = (_idx >= dx || threadIdx.y >= dy);
    if (!blx){pmx[threadIdx.x][threadIdx.y] = pmu[_idx][threadIdx.y];}
    __syncthreads(); 

    if (blx){return;}
    else if (threadIdx.y == 0){pmc[_idx][threadIdx.y] = px_(pmx[threadIdx.x][0], pmx[threadIdx.x][2]);}
    else if (threadIdx.y == 1){pmc[_idx][threadIdx.y] = py_(pmx[threadIdx.x][0], pmx[threadIdx.x][2]);}
    else if (threadIdx.y == 2){pmc[_idx][threadIdx.y] = pz_(pmx[threadIdx.x][1], pmx[threadIdx.x][3]);}
    else { pmc[_idx][threadIdx.y] = pmx[threadIdx.x][threadIdx.y]; } // Pass Energy through
}







template <typename scalar_t, size_t size_x>
__global__ void PtEtaPhiEK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu, 
    const unsigned int dx, const unsigned int dy
){
    __shared__ double pmx[size_x][4]; 
    __shared__ double pmf[size_x][4]; 
    
    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    const unsigned int _ix = threadIdx.x; 
    const unsigned int _iy = threadIdx.y; 

    const bool blx = (_idx >= dx || _iy >= dy); 
    if (!blx){ pmx[_ix][_iy] = pmc[_idx][_iy]; }
    pmf[_ix][_iy] = pmx[_ix][_iy] * pmx[_ix][_iy]; 
    __syncthreads(); 
    if (blx){return;}
    if (_iy == 0){pmu[_idx][_iy] = p2t_(pmf[_ix][0], pmf[_ix][1]); return;}
    if (_iy == 1){pmu[_idx][_iy] = eta_(pmx[_ix][2], pmx[_ix][3]); return;}
    if (_iy == 2){pmu[_idx][_iy] = phi_(pmx[_ix][0], pmx[_ix][1]); return;}
    pmu[_idx][_iy] = pmx[_ix][_iy];
} 


