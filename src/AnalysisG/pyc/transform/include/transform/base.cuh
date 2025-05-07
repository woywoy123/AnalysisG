#include <utils/atomic.cuh>

///< Template for a CUDA kernel to calculate Px (x-component of momentum).
///< @tparam scalar_t The data type of the tensor elements (e.g., float, double).
///< @param pt Input tensor of transverse momentum values.
///< @param phi Input tensor of azimuthal angle values.
///< @param px Output tensor for Px values.
template <typename scalar_t>
__global__ void PxK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px
){
    ///< Calculate the global thread index.
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    ///< Compute Px using the provided pt and phi values.
    px[idx][0] = px_(&pt[idx][0], &phi[idx][0]);
}

///< Template for a CUDA kernel to calculate Py (y-component of momentum).
///< @tparam scalar_t The data type of the tensor elements.
///< @param pt Input tensor of transverse momentum values.
///< @param phi Input tensor of azimuthal angle values.
///< @param py Output tensor for Py values.
template <typename scalar_t>
__global__ void PyK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py
){
    ///< Calculate the global thread index.
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    ///< Compute Py using the provided pt and phi values.
    py[idx][0] = py_(&pt[idx][0], &phi[idx][0]);
}

///< Template for a CUDA kernel to calculate Pz (z-component of momentum).
///< @tparam scalar_t The data type of the tensor elements.
///< @param pt Input tensor of transverse momentum values.
///< @param eta Input tensor of pseudorapidity values.
///< @param pz Output tensor for Pz values.
template <typename scalar_t>
__global__ void PzK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> eta, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pz
){
    ///< Calculate the global thread index.
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    ///< Compute Pz using the provided pt and eta values.
    pz[idx][0] = pz_(&pt[idx][0], &eta[idx][0]);
} 

///< Template for a CUDA kernel to calculate Px, Py, and Pz components of momentum.
///< @tparam scalar_t The data type of the tensor elements.
///< @param pmu Input tensor of momentum components.
///< @param pmc Output tensor for calculated momentum components.
template <typename scalar_t>
__global__ void PxPyPzK(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc
){
    ///< Shared memory for intermediate calculations.
    extern __shared__ double pmx[]; 

    ///< Calculate the global thread index.
    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    ///< Load input data into shared memory.
    pmx[threadIdx.y] = pmu[_idx][threadIdx.y]; 
    __syncthreads(); 
    ///< Perform calculations based on thread index.
    if (threadIdx.y == 0){pmc[_idx][threadIdx.y] = px_(&pmx[0], &pmx[2]); return;}
    if (threadIdx.y == 1){pmc[_idx][threadIdx.y] = py_(&pmx[0], &pmx[2]); return;}
    if (threadIdx.y == 2){pmc[_idx][threadIdx.y] = pz_(&pmx[0], &pmx[1]); return;}
    pmc[_idx][threadIdx.y] = pmx[threadIdx.y]; 
}

///< Template for a CUDA kernel to calculate Px, Py, Pz, and E components of momentum.
///< @tparam scalar_t The data type of the tensor elements.
///< @param pmu Input tensor of momentum components.
///< @param pmc Output tensor for calculated momentum components.
template <typename scalar_t>
__global__ void PxPyPzEK(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu,
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc
){
    ///< Shared memory for intermediate calculations.
    __shared__ double pmx[4]; 
    __shared__ double pmt[4]; 

    ///< Calculate the global thread index.
    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    ///< Load input data into shared memory.
    if (threadIdx.y < 3){pmx[threadIdx.y] = pmu[_idx][threadIdx.y];}
    else {pmt[threadIdx.y] = 0;}

    __syncthreads(); 
    ///< Perform calculations based on thread index.
    double xt = 0; 
    if (threadIdx.y == 0){xt = px_(&pmx[0], &pmx[2]);}
    if (threadIdx.y == 1){xt = py_(&pmx[0], &pmx[2]);}
    if (threadIdx.y == 2){xt = pz_(&pmx[0], &pmx[1]);}
    pmt[threadIdx.y] = xt*xt;
    __syncthreads(); 

    if (threadIdx.y == 3){xt = _sqrt(_sum(pmt, 3));} 
    pmc[_idx][threadIdx.y] = xt; 
}

///< Template for a CUDA kernel to calculate Pt (transverse momentum).
///< @tparam scalar_t The data type of the tensor elements.
///< @param px Input tensor of x-component of momentum values.
///< @param py Input tensor of y-component of momentum values.
///< @param pt Output tensor for Pt values.
template <typename scalar_t> 
__global__ void PtK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pt
){
    ///< Calculate the global thread index.
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    ///< Compute Pt using the provided px and py values.
    pt[idx][0] = pt_(&px[idx][0], &py[idx][0]);
}

///< Template for a CUDA kernel to calculate Phi (azimuthal angle).
///< @tparam scalar_t The data type of the tensor elements.
///< @param px Input tensor of x-component of momentum values.
///< @param py Input tensor of y-component of momentum values.
///< @param phi Output tensor for Phi values.
template <typename scalar_t> 
__global__ void PhiK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> px, 
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> py, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi
){
    ///< Calculate the global thread index.
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    ///< Compute Phi using the provided px and py values.
    phi[idx][0] = phi_(&px[idx][0], &py[idx][0]);
}        

///< Template for a CUDA kernel to calculate Eta (pseudorapidity).
///< @tparam scalar_t The data type of the tensor elements.
///< @param pmc Input tensor of momentum components.
///< @param eta Output tensor for Eta values.
template <typename scalar_t> 
__global__ void EtaK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> eta
){
    ///< Calculate the global thread index.
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    ///< Compute Eta using the provided pmc values.
    eta[idx][0] = eta_(&pmc[idx][0], &pmc[idx][1], &pmc[idx][2]);
}

///< Template for a CUDA kernel to calculate Pt, Eta, and Phi components of momentum.
///< @tparam scalar_t The data type of the tensor elements.
///< @param pmc Input tensor of momentum components.
///< @param pmu Output tensor for calculated momentum components.
template <typename scalar_t>
__global__ void PtEtaPhiK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu
){
    ///< Shared memory for intermediate calculations.
    extern __shared__ double pmx[]; 

    ///< Calculate the global thread index.
    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    ///< Load input data into shared memory.
    pmx[threadIdx.y] = pmc[_idx][threadIdx.y]; 
    __syncthreads(); 

    ///< Perform calculations based on thread index.
    double rx = 0; 
    if (threadIdx.y == 0){rx = pt_(&pmx[0], &pmx[1]);}
    else if (threadIdx.y == 1){
        rx = pt_(&pmx[0], &pmx[1]); 
        rx = eta_(&rx, &pmx[2]); 
    }
    else if (threadIdx.y == 2){rx = phi_(&pmx[0], &pmx[1]);}
    else {rx = pmx[threadIdx.y];}
    pmu[_idx][threadIdx.y] = rx; 
} 

///< Template for a CUDA kernel to calculate Pt, Eta, Phi, and E components of momentum.
///< @tparam scalar_t The data type of the tensor elements.
///< @param pmc Input tensor of momentum components.
///< @param pmu Output tensor for calculated momentum components.
template <typename scalar_t>
__global__ void PtEtaPhiEK(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmu
){
    ///< Shared memory for intermediate calculations.
    __shared__ double pmx[4]; 
    __shared__ double pmt[4]; 

    ///< Calculate the global thread index.
    const unsigned int _idx = blockIdx.x*blockDim.x + threadIdx.x; 
    ///< Load input data into shared memory.
    double rx = 0; 
    if (threadIdx.y < 3){rx = pmc[_idx][threadIdx.y];}
    pmx[threadIdx.y] = rx; 
    pmt[threadIdx.y] = rx*rx;  
    __syncthreads(); 

    ///< Perform calculations based on thread index.
    if (threadIdx.y == 0){rx =  pt_(&pmx[0], &pmx[1]);}
    else if (threadIdx.y == 1){
        rx = pt_(&pmx[0], &pmx[1]); 
        rx = eta_(&rx, &pmx[2]); 
    }
    else if (threadIdx.y == 2){rx = phi_(&pmx[0], &pmx[1]);}
    else {rx = _sqrt(_sum(pmt, 3));}
    pmu[_idx][threadIdx.y] = rx; 
}

