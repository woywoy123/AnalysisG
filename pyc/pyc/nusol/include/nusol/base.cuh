#ifndef CU_NUSOL_BASE_H
#define CU_NUSOL_BASE_H
#include <torch/torch.h>
#include <atomic/cuatomic.cuh>

struct nusol {
    double cos, sin;
    double x0, x0p; 
    double sx, sy;
    double w, w_; 
    double x1, y1; 
    double z, o2, eps2; 

    // index: 0 -> lepton, 1 -> b-quark
    double pmass[2]  = {0x0};
    double betas[2]  = {0x0};
    double pmu_b[4]  = {0x0};
    double pmu_l[4]  = {0x0};
    double masses[3] = {0x0}; 
    bool passed = true; 
    nusol() = default; 
}; 


template <typename scalar_t>
__global__ void _shape_matrix(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out,
        unsigned int dx, unsigned int dy, unsigned int dl, long* diag
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    if (_idx >= dx || _idy >= dy || _idz >= dl){return;}
    out[_idx][_idy][_idz] += diag[_idz];  
}


namespace nusol_ {
    torch::Tensor ShapeMatrix(torch::Tensor* inpt, std::vector<long> vec); 
}

__device__ void _makeNuSol(nusol* sl){
    double b2l = sl -> betas[0]; 
    double b2b = sl -> betas[1]; 

    for (size_t x(0); x < 3; ++x){sl -> masses[x] = pow(sl -> masses[x], 2);}
    for (size_t x(0); x < 2; ++x){sl -> betas[x] = _sqrt(&sl -> betas[x]);}
    sl -> eps2 = (sl -> masses[1] - sl -> masses[2])*(1 - b2b); 

    sl -> sin = 1 - pow(sl -> cos, 2); 
    sl -> sin = _sqrt(&sl -> sin); 
    double div_sin = _div(&sl -> sin); 

    double r = sl -> betas[0] * _div(&sl -> betas[1]); 
    sl -> w    = ( r - sl -> cos) * div_sin; 
    sl -> w_   = (-r - sl -> cos) * div_sin; 
    sl -> o2   = pow(sl -> w, 2) + 1 - b2l; 
    double _div_o2 = _div(&sl -> o2); 

    sl -> x0  = - (sl -> masses[1] - sl -> masses[2] - sl -> pmass[0]) * _div(&sl -> pmu_l[3]) * 0.5;
    sl -> x0p = - (sl -> masses[0] - sl -> masses[1] - sl -> pmass[1]) * _div(&sl -> pmu_b[3]) * 0.5;

    sl -> sx = (sl -> x0 * sl -> betas[0] - sl -> betas[0] * sl -> pmu_l[3] * (1 - b2l)) * _div(&b2l); 
    sl -> sy = (sl -> x0p * _div(&sl -> betas[1]) - sl -> cos * sl -> sx) * div_sin; 
 
    sl -> x1 = sl -> sx - (sl -> sx + sl -> w * sl -> sy) * _div_o2; 
    sl -> y1 = sl -> sy - (sl -> sx + sl -> w * sl -> sy) * sl -> w * _div_o2; 
    sl -> passed *= (_div_o2 > 0) * (r > 0); 

    double z2 = pow(sl -> x1, 2)*sl -> o2; 
    z2 -= pow(sl -> sy - sl -> w * sl -> sx, 2); 
    z2 -= (sl -> masses[1] - pow(sl -> x0, 2) - sl -> eps2); 
    sl -> z = (z2 <= 0) ? 0 : _sqrt(&z2); 
}


__device__ double _krot(nusol* sl, const unsigned int _iky, const unsigned int _ikz){
    double r = ((_ikz < 2)*(_iky < 2) + (_ikz == _iky)) > 0; 
    r *= (_ikz == _iky)*sl -> cos + (_iky - _ikz)*sl -> sin;
    return r; 
}

__device__ double _amu(nusol* sl, const unsigned int _iky, const unsigned int _ikz){
    bool ii = (_ikz == _iky); 

    double b2 = pow(sl -> betas[0], 2);
    double val = ii*(_iky == 0)*(1 - b2);
    val += ii*(_iky == 3)*(sl -> masses[1] - pow(sl -> x0, 2) - sl -> eps2); 
    val += ii*(_ikz == 2 + _ikz == 1); 
    val += ((_ikz == 3)*(_iky == 0) + (_ikz == 0)*(_iky == 3))*(sl -> sx * b2); 
    return val; 
}

__device__ double _abq(nusol* sl, const unsigned int _iky, const unsigned int _ikz){
    bool ii = (_ikz == _iky); 

    double val = ii*(_iky == 0)*(1 - pow(sl -> betas[1], 2));
    val += ii*(_iky == 3)*(sl -> masses[1] - pow(sl -> x0p, 2)); 
    val += ii*(_ikz == 2 + _ikz == 1); 
    val += ((_ikz == 3)*(_iky == 0) + (_ikz == 0)*(_iky == 3))*(sl -> betas[1] * sl -> x0p); 
    return val; 
}


// ---- Htilde -----
// [Z/o, 0, x1 - P_l], [w * Z / o, 0, y1], [0, Z, 0]
__device__ double _htilde(nusol* sl, const unsigned int _iky, const unsigned int _ikz){
    double z_div_o = _sqrt( &sl -> o2 ); 
    z_div_o = (sl -> z) * _div(&z_div_o); 

    double val = (_ikz == _iky) * (_ikz == 0) * z_div_o;  // Z / o
    val += (_iky == 1) * (_ikz == 0) * z_div_o * sl -> w; // w * Z / o
    val += (_iky == 2) * (_ikz == 1) * sl -> z;           // Z
    val += (_iky == 0) * (_ikz == 2) * (sl -> x1 - sl -> betas[0] * sl -> pmu_l[3]); // x1 - P_l
    val += (_iky == 1) * (_ikz == 2) * (sl -> y1);
    return val;  
}





template <typename scalar_t>
__global__ void _hmatrix(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> masses, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> cosine, 

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_l, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2l, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2l, 

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2b, 
        const unsigned int dx, nusol* solx
){
    extern __shared__ nusol smem[]; 
    __shared__ double K[4][4]; 
    __shared__ double KT[4][4]; 
    __shared__ double Kdot[4][4]; 

    __shared__ double A_l[4][4];
    __shared__ double A_b[4][4]; 
    __shared__ double Htil[4][4]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _iky = threadIdx.y;
    const unsigned int _ikz = threadIdx.z; 

    nusol* sl = nullptr; 
    if (!threadIdx.x && !threadIdx.y && !threadIdx.z){
        sl = &smem[0]; 
        sl -> cos = cosine[_idx][0]; 
        sl -> betas[0] = b2l[_idx][0]; 
        sl -> pmass[0] = m2l[_idx][0]; 

        sl -> betas[1] = b2b[_idx][0]; 
        sl -> pmass[1] = m2b[_idx][0]; 
        for (size_t x(0); x < 3; ++x){sl -> masses[x] = masses[_idx][x];} 
        for (size_t x(0); x < 4; ++x){sl -> pmu_b[x] = pmc_b[_idx][x];}
        for (size_t x(0); x < 4; ++x){sl -> pmu_l[x] = pmc_l[_idx][x];}
        _makeNuSol(sl); 
    }
    __syncthreads(); 
    sl = &smem[0]; 

    K[_iky][_ikz]    =   _krot(sl, _iky, _ikz) * sl -> passed;
    A_l[_iky][_ikz]  =    _amu(sl, _iky, _ikz) * sl -> passed;  
    A_b[_iky][_ikz]  =    _abq(sl, _iky, _ikz) * sl -> passed; 
    Htil[_iky][_ikz] = _htilde(sl, _iky, _ikz) * sl -> passed;  
    KT[_iky][_ikz]   = K[_ikz][_iky];  
    __syncthreads(); 
    for (size_t x(0); x < 4; ++x){Kdot[_iky][_ikz] += K[_iky][x] * A_b[x][_ikz];}
    __syncthreads();

    A_b[_iky][_ikz] = 0; 
    for (size_t x(0); x < 4; ++x){A_b[_iky][_ikz] += Kdot[_iky][x] * KT[x][_ikz];}



}



#endif
