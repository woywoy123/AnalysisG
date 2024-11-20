#include <atomic/cuatomic.cuh>

template <typename scalar_t>
__global__ void PxK(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> phi, 
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> px,
    const unsigned int dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < dx){ px[idx] = px_(&pt[idx], &phi[idx]); }
}

template <typename scalar_t>
__global__ void PyK(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> phi, 
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> py,
    const unsigned int dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < dx){ py[idx] = py_(&pt[idx], &phi[idx]); }
}

template <typename scalar_t>
__global__ void PzK(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> eta, 
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> pz,
    const unsigned int dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < dx){ pz[idx] = pz_(&pt[idx], &eta[idx]); }
} 

template <typename scalar_t>
__global__ void PxPyPzK(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        const unsigned int dx
){
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    if ((idx >= dx) + (idy >= 3)){return;}
    else if (idy == 0){pmc[idx][idy] = px_(&pmu[idx][0], &pmu[idx][2]);}
    else if (idy == 1){pmc[idx][idy] = py_(&pmu[idx][0], &pmu[idx][2]);}
    else {pmc[idx][idy] = pz_(&pmu[idx][0], &pmu[idx][1]);}
}

template <typename scalar_t>
__global__ void PxPyPzEK(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc,
        const unsigned int dx
){
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    if ((idx >= dx) + (idy >= 4)){return;}
    else if (idy == 0){pmc[idx][idy] = px_(&pmu[idx][0], &pmu[idx][2]);}
    else if (idy == 1){pmc[idx][idy] = py_(&pmu[idx][0], &pmu[idx][2]);}
    else if (idy == 2){pmc[idx][idy] = pz_(&pmu[idx][0], &pmu[idx][1]);}
    else {pmc[idx][idy] = pmu[idx][idy];}
}


template <typename scalar_t> 
__global__ void PtK(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> px, 
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> py, 
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> pt, 
    const unsigned int dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < dx){ pt[idx] = pt_(&px[idx], &py[idx]); }
}

template <typename scalar_t> 
__global__ void EtaK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> eta, 
    const unsigned int dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < dx){ eta[idx] = eta_(&pmc[idx][0], &pmc[idx][1], &pmc[idx][2]); }
}

template <typename scalar_t> 
__global__ void PhiK(
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> px, 
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> py, 
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> phi, 
    const unsigned int dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < dx){ phi[idx] = phi_( &px[idx], &py[idx]); }
}        

template <typename scalar_t>
__global__ void PtEtaPhiK(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
    const unsigned int dx)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if ((idx >= dx) + (blockIdx.y > 1)){return;}
    if (blockIdx.y == 0){pmu[idx][2] = phi_(&pmc[idx][0], &pmc[idx][1]); return;}
    scalar_t data = pt_(&pmc[idx][0], &pmc[idx][1]); 
    pmu[idx][1] = eta_(&data, &pmc[idx][2]); 
    pmu[idx][0] = data;
} 

template <typename scalar_t>
__global__ void PtEtaPhiEK(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
    const unsigned int dx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    if ((idx >= dx) + (idy > 2)){return;}
    if (idy == 0){pmu[idx][2] = phi_(&pmc[idx][0], &pmc[idx][1]); return;}
    else if (idy == 1){pmu[idx][3] = pmc[idx][3]; return; }
    scalar_t data = pt_(&pmc[idx][0], &pmc[idx][1]);
    pmu[idx][1]   = eta_(&data, &pmc[idx][2]); 
    pmu[idx][0]   = data;
}

