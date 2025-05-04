#include <utils/atomic.cuh>

template <typename scalar_t>
__global__ void PxK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    px[idx][0] = px_(&pt[idx][0], &phi[idx][0]);
}

template <typename scalar_t>
__global__ void PyK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    py[idx][0] = py_(&pt[idx][0], &phi[idx][0]);
}

template <typename scalar_t>
__global__ void PzK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> eta, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pz
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    pz[idx][0] = pz_(&pt[idx][0], &eta[idx][0]);
} 

template <typename scalar_t, size_t size_x>
__global__ void PxPyPzK(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int dx, const unsigned int dy
){
    __shared__ double pmx[size_x][4]; 

    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    pmx[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}
    pmx[threadIdx.x][threadIdx.y] = pmu[_idx][threadIdx.y]; 
    __syncthreads(); 

    if (threadIdx.y == 0){pmc[_idx][threadIdx.y] = px_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][2]); return;}
    if (threadIdx.y == 1){pmc[_idx][threadIdx.y] = py_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][2]); return;}
    if (threadIdx.y == 2){pmc[_idx][threadIdx.y] = pz_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][1]); return;}
    pmc[_idx][threadIdx.y] = pmx[threadIdx.x][threadIdx.y]; 
}

template <typename scalar_t, size_t size_x>
__global__ void PxPyPzEK(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        const unsigned int dx, const unsigned int dy
){
    __shared__ double pmx[size_x][4]; 
    __shared__ double pmt[size_x][4]; 

    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    pmx[threadIdx.x][threadIdx.y] = 0;
    pmt[threadIdx.x][threadIdx.y] = 0;
    if (_idx >= dx || threadIdx.y >= dy){return;}
    pmx[threadIdx.x][threadIdx.y] = pmu[_idx][threadIdx.y]; 
    __syncthreads(); 

    double xt = 0; 
    if (threadIdx.y == 0){xt = px_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][2]);}
    if (threadIdx.y == 1){xt = py_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][2]);}
    if (threadIdx.y == 2){xt = pz_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][1]);}
    pmt[threadIdx.x][threadIdx.y] = xt*xt;
    __syncthreads(); 

    if (threadIdx.y == 3){xt = _sqrt(_sum(pmt[threadIdx.x], 3));} 
    pmc[_idx][threadIdx.y] = xt; 
}


template <typename scalar_t> 
__global__ void PtK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    pt[idx][0] = pt_(&px[idx][0], &py[idx][0]);
}

template <typename scalar_t> 
__global__ void PhiK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    phi[idx][0] = phi_(&px[idx][0], &py[idx][0]);
}        


template <typename scalar_t> 
__global__ void EtaK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> eta
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    eta[idx][0] = eta_(&pmc[idx][0], &pmc[idx][1], &pmc[idx][2]);
}

template <typename scalar_t, size_t size_x>
__global__ void PtEtaPhiK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu, 
    const unsigned int dx, const unsigned int dy
){
    __shared__ double pmx[size_x][4]; 
    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    pmx[threadIdx.x][threadIdx.y] = 0; 
    if (_idx >= dx || threadIdx.y >= dy){return;}
    pmx[threadIdx.x][threadIdx.y] = pmc[_idx][threadIdx.y]; 
    __syncthreads(); 

    double rx = 0; 
    if (threadIdx.y == 0){rx = pt_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][1]);}
    else if (threadIdx.y == 1){
        rx = pt_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][1]); 
        rx = eta_(&rx, &pmx[threadIdx.x][2]); 
    }
    else if (threadIdx.y == 2){rx = phi_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][1]);}
    else {rx = pmx[threadIdx.x][threadIdx.y];}
    pmu[_idx][threadIdx.y] = rx; 
} 

template <typename scalar_t, size_t size_x>
__global__ void PtEtaPhiEK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu, 
    const unsigned int dx, const unsigned int dy
){
    __shared__ double pmx[size_x][4]; 
    __shared__ double pmt[size_x][4]; 

    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    if (_idx >= dx){return;}

    double rx = 0; 
    if (threadIdx.y < 3){rx = pmc[_idx][threadIdx.y];}
    pmx[threadIdx.x][threadIdx.y] = rx; 
    pmt[threadIdx.x][threadIdx.y] = rx*rx;  
    __syncthreads(); 

    if (threadIdx.y == 0){rx =  pt_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][1]);}
    else if (threadIdx.y == 1){
        rx = pt_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][1]); 
        rx = eta_(&rx, &pmx[threadIdx.x][2]); 
    }
    else if (threadIdx.y == 2){rx = phi_(&pmx[threadIdx.x][0], &pmx[threadIdx.x][1]);}
    else {rx = _sqrt(_sum(pmt[threadIdx.x], 3));}
    pmu[_idx][threadIdx.y] = rx; 
}

