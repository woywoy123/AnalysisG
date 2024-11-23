#ifndef CU_NUSOL_BASE_H
#define CU_NUSOL_BASE_H

#include <torch/torch.h>
#include <atomic/cuatomic.cuh>

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

    sl -> x0  = - (sl -> masses[1] - sl -> masses[2] - sl -> pmass[0]) * _div(&sl -> pmu_l[3]) * 0.5;
    sl -> x0p = - (sl -> masses[0] - sl -> masses[1] - sl -> pmass[1]) * _div(&sl -> pmu_b[3]) * 0.5;

    sl -> sx = (sl -> x0 * sl -> betas[0] - sl -> betas[0] * sl -> pmu_l[3] * (1 - b2l)) * _div(&b2l); 
    sl -> sy = (sl -> x0p * _div(&sl -> betas[1]) - sl -> cos * sl -> sx) * div_sin; 
 
    double _div_o2 = _div(&sl -> o2); 
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
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> rt,

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_l, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2l, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2l, 

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2b, 

        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Hmatrix,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp ,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> KMatrix,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A_leps,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A_bqrk,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> isNan
){
    __shared__ double K[4][4]; 
    __shared__ double KT[4][4]; 

    __shared__ double Kdot[4][4]; 
    __shared__ double rotT[3][3]; 

    __shared__ double A_l[4][4];
    __shared__ double A_b[4][4]; 

    __shared__ double Htil[4][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    const unsigned int _iky = (_idy * 4 + _idz)/4; 
    const unsigned int _ikz = (_idy * 4 + _idz)%4; 

    nusol sl = nusol(); 
    sl.cos = cosine[_idx][0]; 
    sl.betas[0] = b2l[_idx][0]; 
    sl.pmass[0] = m2l[_idx][0]; 

    sl.betas[1] = b2b[_idx][0]; 
    sl.pmass[1] = m2b[_idx][0]; 
    for (size_t x(0); x < 4; ++x){
        sl.pmu_b[x] = pmc_b[_idx][x];
        sl.pmu_l[x] = pmc_l[_idx][x];
        if (x > 2){continue;}
        sl.masses[x] = masses[_idx][x];
    } 
    _makeNuSol(&sl); 
    if (_iky < 3 && _ikz < 3){rotT[_iky][_ikz] = rt[_idx][_iky][_ikz];}

    Htil[_iky][_ikz] = _htilde(&sl, _iky, _ikz);  
    A_l[_iky][_ikz]  = _amu(&sl, _iky, _ikz);  
    A_b[_iky][_ikz]  = _abq(&sl, _iky, _ikz); 

    double rotx = _krot(&sl, _iky, _ikz);
    K[_iky][_ikz]  = rotx; 
    KT[_ikz][_iky] = rotx; 
    KMatrix[_iky][_iky][_ikz] = rotx; 
    __syncthreads(); 

    if (_iky < 3 && _ikz < 3){
        double hx = _dot(rotT, Htil, _iky, _ikz, 3); 
        Hmatrix[_idx][_iky][_ikz] = hx;
        H_perp[_idx][_iky][_ikz] = (_iky < 2) ? _ikz == 2 : hx; 
    }

    Kdot[_iky][_ikz] = _dot(K, A_b, _iky, _ikz, 4); 
    __syncthreads();

    A_b[_iky][_ikz] = _dot(Kdot, KT, _iky, _ikz, 4); 
    A_leps[_iky][_iky][_ikz] = A_l[_iky][_ikz]; 
    A_bqrk[_iky][_iky][_ikz] = A_b[_iky][_ikz]; 
    isNan[_idx] = sl.passed; 
}

template <typename scalar_t>
__global__ void _nu_init_(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> s2,
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> met_xy,
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H, 

        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> M
){

    __shared__ double _H[3][3];
    __shared__ double _S2[3][3]; 
    __shared__ double _V0[3][3]; 

    __shared__ double _dNu[3][3]; 
    __shared__ double _dNuT[3][3]; 

    __shared__ double _X[3][3]; 
    __shared__ double _T[3][3]; 
    __shared__ double _Dx[3][3]; 

    __shared__ double _XD[3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = threadIdx.y;
    const unsigned int _idz = threadIdx.z;  

    // ------- Populate data ----------- //
    _S2[_idy][_idz] = 0; 
    _V0[_idy][_idz] = 0; 
    _H[_idy][_idz] = H[_idx][_idy][_idz]; 

    double pi = M_PI*0.5; 
    _Dx[_idy][_idz] = _rz(&pi, _idy, _idz); 
    _T[_idy][_idz] = (_idy == _idz)*(_idy < 2); 

    if (_idy < 2  && _idz < 2){_S2[_idy][_idz] = s2[_idx][_idy][_idz];}
    if (_idz == 2 && _idy < 2){_V0[_idy][_idz] = met_xy[_idx][_idy];}
    __syncthreads(); 

    // ------- matrix inversion for S2 ------ //
    if (!_idy && !_idz){
        double s00 = _S2[0][0]; 
        double s11 = _S2[1][1]; 
        double s01 = _S2[0][1]; 
        double s10 = _S2[1][0]; 
        double det = (s00*s11 - s01*s10);
        det = _div(&det);
    
        // S2^-1 with transpose
        _S2[0][0] =  s11*det; 
        _S2[1][1] =  s00*det; 

        _S2[0][1] = -s10*det; 
        _S2[1][0] = -s01*det; 
    }

    double di = _dot(_Dx, _T, _idy, _idz, 3); 
    _dNu[_idy][_idz]  = _V0[_idy][_idz] - _H[_idy][_idz]; 
    _dNuT[_idz][_idy] = _V0[_idy][_idz] - _H[_idy][_idz]; 
    __syncthreads(); 

    _Dx[_idy][_idz] = di; 
    _T[_idy][_idz] = _dot(_dNuT, _S2, _idy, _idz, 3); 
    __syncthreads(); 

    _X[_idy][_idz] = _dot(_T, _dNu, _idy, _idz, 3); 
    __syncthreads(); 

    _T[_idy][_idz]  = (_idy == _idz)*(2*(_idy < 2) - 1); 
    _XD[_idy][_idz] = _dot(_X, _Dx, _idy, _idz, 3) + _dot(_X, _Dx, _idz, _idy, 3);  

    X[_idx][_idy][_idz] = _T[_idy][_idz];
    M[_idx][_idy][_idz] = _XD[_idy][_idz]; 
}


template <typename scalar_t>
__global__ void _swapAB(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> inv_A_dot_B,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> B,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> detA, 
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> detB
){

    __shared__ double A_[3][3]; 
    __shared__ double B_[3][3]; 

    __shared__ double _cofA[3][3];  
    __shared__ double _detA[3][3];
    __shared__ double _InvA[3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    A_[threadIdx.y][threadIdx.z]  = A[_idx][threadIdx.y][threadIdx.z]; 
    B_[threadIdx.y][threadIdx.z]  = B[_idx][threadIdx.y][threadIdx.z]; 

    // ----- swap if abs(det(B)) > abs(det(A)) -------- //
    bool swp = abs(detB[_idx][0]) > abs(detA[_idx][0]); 
    double a_ = (!swp)*A_[threadIdx.y][threadIdx.z] + (swp)*B_[threadIdx.y][threadIdx.z]; 
    double b_ = (!swp)*B_[threadIdx.y][threadIdx.z] + (swp)*A_[threadIdx.y][threadIdx.z];
    A_[threadIdx.y][threadIdx.z] = a_;  
    B_[threadIdx.y][threadIdx.z] = b_;  
    __syncthreads(); 

    // ----- compute the inverse of A -------- //
    double mx = _cofactor(A_, threadIdx.y, threadIdx.z); 
    _cofA[threadIdx.z][threadIdx.y] = mx;  // transpose cofactor matrix to get adjoint 
    _detA[threadIdx.y][threadIdx.z] = mx * A_[threadIdx.y][threadIdx.z]; 
    __syncthreads(); 

    double _dt = _detA[0][0] + _detA[0][1] + _detA[0][2];
    _InvA[threadIdx.y][threadIdx.z] = _cofA[threadIdx.y][threadIdx.z]*_div(&_dt); 
    __syncthreads(); 
    // --------------------------------------- //

    // -------- take the dot product of inv(A) and B --------- //
    inv_A_dot_B[_idx][threadIdx.y][threadIdx.z] = _dot(_InvA, B_, threadIdx.y, threadIdx.z, 3); 
    B[_idx][threadIdx.y][threadIdx.z] = B_[threadIdx.y][threadIdx.z]; 
    A[_idx][threadIdx.y][threadIdx.z] = A_[threadIdx.y][threadIdx.z]; 
} 



__device__ double _case1(double G[3][3], const unsigned int _idy, const unsigned int _idz){
    if (_idy == 0 && _idz == 0){return G[0][1];}
    if (_idy == 0 && _idz == 2){return G[1][2];}
    if (_idy == 1 && _idz == 1){return G[0][1];}
    if (_idy == 1 && _idz == 2){return G[0][2] - G[1][2];}
    return 0; 
}

__device__ double _case2(double G[3][3], const unsigned int _idy, const unsigned int _idz, bool swpXY){
    if (!swpXY){return G[_idy][_idz];}
    if (_idy == 0 && _idz == 0){return G[1][1];}
    if (_idy == 0 && _idz == 1){return G[1][0];}
    if (_idy == 0 && _idz == 2){return G[1][2];}

    if (_idy == 1 && _idz == 0){return G[0][1];}
    if (_idy == 1 && _idz == 1){return G[0][0];}
    if (_idy == 1 && _idz == 2){return G[0][2];}

    if (_idy == 2 && _idz == 0){return G[2][1];}
    if (_idy == 2 && _idz == 1){return G[2][0];}
    if (_idy == 2 && _idz == 2){return G[2][2];}
    return 0; 
}

__device__ double _leqnulls(double coF[3][3], double Q[3][3], const unsigned int _idy, const unsigned int _idz){
    if (_idy >= 2){return 0;}
    double q00 = -coF[0][0]; 
    if (q00 < 0){return 0;}

    if (_idy == 0 && _idz == 0){return Q[0][1];}
    if (_idy == 0 && _idz == 1){return Q[1][1];}
    if (_idy == 0 && _idz == 2){return Q[1][2] - _sqrt(&q00);}
    if (q00 == 0){return 0;}

    if (_idy == 1 && _idz == 0){return Q[0][1];}
    if (_idy == 1 && _idz == 1){return Q[1][1];}
    if (_idy == 1 && _idz == 2){return Q[1][2] + _sqrt(&q00);}
    return 0; 
}

__device__ double _gnulls(double coF[3][3], double Q[3][3], const unsigned int _idy, const unsigned int _idz){
    if (_idy >= 2){return 0;}
    double s22 = _sqrt(-coF[2][2]);
    double q22 = _div(coF[2][2]); 

    if (_idy == 0 && _idz == 0){return  Q[0][1] - s22;}
    if (_idy == 0 && _idz == 1){return  Q[1][1];}
    if (_idy == 0 && _idz == 2){return -Q[1][1] * coF[1][2] * q22 - (Q[0][1] - s22) * coF[0][2] * q22;}

    if (_idy == 1 && _idz == 0){return  Q[0][1] + s22;}
    if (_idy == 1 && _idz == 1){return  Q[1][1];}
    if (_idy == 1 && _idz == 2){return -Q[1][1] * coF[1][2] * q22 - (Q[0][1] + s22) * coF[0][2] * q22;}
    return 0; 
}

template <typename scalar_t>
__global__ void _factor_degen(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> real,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> imag,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> B,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> Lins,
        const double nulls
){
    __shared__ double G[3][3][3]; 
    __shared__ double g[3][3][3]; 
    __shared__ double coG[3][3][3]; 
    __shared__ double lines[3][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _ixd = threadIdx.y*9 + threadIdx.z; 
    const unsigned int _idt = (_ixd / 9)%3; 
    const unsigned int _idy = (_ixd / 3)%3; 
    const unsigned int _idz =  _ixd % 3; 
    if (abs(imag[_idx][_idt]) > 0.0){return;}

    G[_idt][_idy][_idz] = B[_idx][_idy][_idz] - real[_idx][_idt] * A[_idx][_idy][_idz]; 
    __syncthreads(); 

    bool c1 = (G[_idt][0][0] == G[_idt][1][1]) * (G[_idt][1][1] == 0); 
    if (c1){Lins[_idx][_idt][_idy][_idz] = _case1(G[_idt], _idy, _idz); return;}

    lines[_idt][_idy][_idz] = 0; 
    bool sw = abs(G[_idt][0][0]) > abs(G[_idt][1][1]); 
    g[_idt][_idy][_idz] = _case2(G[_idt], _idy, _idz, sw)*_div(G[_idt][!sw][!sw]);
    __syncthreads();

    coG[_idt][_idy][_idz] = _cofactor(g[_idt], _idy, _idz); 
    __syncthreads(); 

    double elx = 0; 
    if (-coG[_idt][2][2] <= nulls){ elx = _leqnulls(coG[_idt], g[_idt], _idy, _idz); }
    else { elx = _gnulls(coG[_idt], g[_idt], _idy, _idz); }

    if (_idz == 0){Lins[_idx][_idt][_idy][1 - !sw] = elx;}
    if (_idz == 1){Lins[_idx][_idt][_idy][!sw] = elx;}
    if (_idz == 2){Lins[_idx][_idt][_idy][2] = elx;}
}




#endif
