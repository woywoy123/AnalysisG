#ifndef CU_NUSOL_BASE_H
#define CU_NUSOL_BASE_H

#include <torch/torch.h>
#include <atomic/cuatomic.cuh>
#include <nusol/device.cuh>

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
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> M, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Unit
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

    X[_idx][_idy][_idz] = _X[_idy][_idz]; 
    M[_idx][_idy][_idz] = _XD[_idy][_idz]; 
    Unit[_idx][_idy][_idz] = _T[_idy][_idz];
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

template <typename scalar_t>
__global__ void _intersections(
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> real,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> imag,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> ellipse,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> lines,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_pts,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_dst,
        const double nulls
){
    __shared__ double _real[9][3][3]; 
    __shared__ double _imag[9][3][3];

    __shared__ double _elip[9][3][3]; 
    __shared__ double _line[9][3][3]; 
    __shared__ double _solx[9][3][3]; 
    __shared__ double _soly[9][3][3]; 
    __shared__ double _dist[9][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = threadIdx.y/9;
    const unsigned int _idz = threadIdx.y/3;
    const unsigned int _idt = threadIdx.y%3; 
    
    _real[_idz][_idt][threadIdx.z] = real[_idx][_idz][_idt][threadIdx.z]; 
    _imag[_idz][_idt][threadIdx.z] = imag[_idx][_idz][_idt][threadIdx.z]; 

    _line[_idz][_idt][threadIdx.z] = lines[_idx][_idy][_idz%3][threadIdx.z];
    _elip[_idz][_idt][threadIdx.z] = ellipse[_idx][_idt][threadIdx.z];
    __syncthreads();  

    double v1 = 0; 
    for (size_t x(0); x < 3; ++x){v1 += _line[_idz][_idt][x] * _real[_idz][_idt][x];}
    _soly[_idz][_idt][threadIdx.z] = _dot(_real[_idz], _elip[_idz], _idt, threadIdx.z, 3); 
    _solx[_idz][_idt][threadIdx.z] = _real[_idz][_idt][threadIdx.z]*_div(_real[_idz][_idt][2]); 
    __syncthreads(); 

    double v2 = 0; 
    for (size_t x(0); x < 3; ++x){v2 += _soly[_idz][_idt][x] * _real[_idz][_idt][x];}
    v2 = (v2 + v1) ? log10(v2*v2 + v1*v1) : 200;

    _dist[_idz][_idt][threadIdx.z] = (v2 < log10(nulls)) ? v2 : 200; 
    s_pts[_idx][_idz][_idt][threadIdx.z] = _dist[_idz][_idt][threadIdx.z]; 
    s_dst[_idx][_idz][_idt][threadIdx.z] = _solx[_idz][_idt][threadIdx.z]; 
}


template <typename scalar_t>
__global__ void _solsx(
        const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_pts,
        const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_dst,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> sols,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> solx,
        torch::PackedTensorAccessor64<long    , 4, torch::RestrictPtrTraits> idxs
){
    __shared__ double _point[9][3][3]; 
    __shared__ double _lines[9][3][3]; 
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    const unsigned int _id1 = _idy/3; 
    const unsigned int _id2 = _idy%3; 
    
    long id_ = idxs[_idx][_id1][_id2][_idz]; 
    _point[_id1][_id2][_idz] = s_pts[_idx][_id1][id_][_idz]; 
    _lines[_id1][_id2][_idz] = s_dst[_idx][_id1][id_][_idz]; 
    __syncthreads(); 

    const unsigned dx_ = _id1*2 + _id2; 
    if (_id2 < 2){sols[_idx][dx_][_idz] = _lines[_id1][_id2][_idz];}
    if (_id2 < 2 && !threadIdx.z){solx[_idx][dx_][0] = _point[_id1][_id2][0];}
}


template <typename scalar_t>
__global__ void _chi2(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> sols,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> dst,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> chi2
){
    __shared__ double _S[18][3]; 
    __shared__ double _X[18][3][3]; 
    __shared__ double _H[18][3][3]; 
    __shared__ double _C[18][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    const bool _dst = dst[_idx][_idy][0] < 200; 

    _S[_idy][_idz] = sols[_idx][_idy][_idz]; 
    for (size_t x(0); x < 3; ++x){_X[_idy][_idz][x] = X[_idx][_idz][x];}
    for (size_t x(0); x < 3; ++x){_H[_idy][_idz][x] = H[_idx][_idz][x];}
    __syncthreads();
    nu[_idx][_idy][_idz] = _dot(_H[_idy][_idz], _S[_idy], 3)*_dst; 
    _C[_idy][_idz]       = _dot(_S, _X[_idy], _idy, _idz, 3); 
    __syncthreads();

    if (_idz){return;}
    chi2[_idx][_idy][0]  = _dot(_C[_idy], _S[_idy], 3)*_dst;
}


#endif
