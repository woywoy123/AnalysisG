#include <torch/torch.h>
#include <cmath>
#include "nusol.cu"
#include "operators.cu"

template <typename scalar_t>
__global__ void _ShapeKernel(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> inpt, 
        const unsigned int len_i, 
        const unsigned int len_k, 
        const unsigned int len_j, 
        const bool assign)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 

    if (idx >= len_i || idy >= len_k || idz >= len_j){ return; }
    if (assign){ out[idx][idy][idz] = inpt[(idx >= inpt.size(0)) ? 0 : idx][idy][idz]; return; }
    if (idy == idz){ out[idx][idy][idz] = inpt[0][0][idz]; }
}

template <typename scalar_t>
__global__ void _H_Base(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> beta2_b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mass2_b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_b, 

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> beta2_mu, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mass2_mu, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_mu, 
 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> cos, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> mass2, 
        const unsigned int dim_i, const unsigned int dim_m)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_i || idy >= 3 || idz >= 3){ return; }    
    if (idy <= 1 && idz == 1){ return; }
    if (idy == 2 && (idz == 0 || idz == 2)){ return; }
    
    scalar_t   mass2_W   = mass2[(idx >= dim_m) ? 0 : idx][0]; 
    scalar_t   mass2_top = mass2[(idx >= dim_m) ? 0 : idx][1]; 
    scalar_t   mass2_nu  = mass2[(idx >= dim_m) ? 0 : idx][2];     
    mass2_W   *= mass2_W;  
    mass2_top *= mass2_top; 
    mass2_nu  *= mass2_nu;  

    scalar_t beta_mu   = sqrt(beta2_mu[idx][0]);
    scalar_t beta_b    = sqrt(beta2_b[idx][0]); 
    scalar_t sin       = sqrt(1 - cos[idx][0]*cos[idx][0]);   
 
    scalar_t x0p  = _x0(mass2_top, mass2_W, mass2_b[idx][0], pmc_b[idx][3]); 
    scalar_t x0   = _x0(mass2_W, mass2_mu[idx][0], mass2_nu, pmc_mu[idx][3]);

    scalar_t Sx   = _Sx(x0, beta2_mu[idx][0], beta_mu, pmc_mu[idx][3]); 
    scalar_t Sy   = _Sy(x0p, Sx, beta_b, cos[idx][0], sin); 

    scalar_t w    = _w(beta_mu , beta_b, cos[idx][0], sin); 
    scalar_t Om2  = _omega2(w, beta2_mu[idx][0]); 

    scalar_t coef = _coef(Sx, Sy, w, Om2); 
    scalar_t x1   = Sx - coef; 
    if (idy == 0 && idz == 2){ out[idx][idy][idz] = x1 - beta_mu * pmc_mu[idx][3]; return; }

    scalar_t y1   = Sy - w*coef; 
    if (idy == 1 && idz == 2){ out[idx][idy][idz] = y1; }

    scalar_t eps2 = _epsilon2(mass2_W, mass2_nu, beta2_mu[idx][0]); 
    scalar_t Z    = _Z(x1, Om2, Sy, Sx, w, mass2_W, x0, eps2); 
    if (idy == 2 && idz == 1){ out[idx][idy][idz] = Z; return; }
    if (idy == 1 && idz == 0){ out[idx][idy][idz] = w*Z/sqrt(Om2); return; }
    if (idy == 0 && idz == 0){ out[idx][idy][idz] = Z/sqrt(Om2); return; }
}

template <typename scalar_t>
__global__ void _Base_Matrix_H_Kernel(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Ry, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Rz, 

        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> RyT, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> RzT, 

        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> theta, 
        const unsigned int dim_x)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_x || idy >= 6 || idz  >= 3){return;}
    if (idy < 3)
    { 
        _rz(Rz[idx][idy][idz], -phi[idx][0], idy, idz); 
        RzT[idx][idz][idy] = Rz[idx][idy][idz]; 
    }
    else
    {
        _pihalf(theta[idx][0]); 
        _ry(Ry[idx][idy%3][idz], theta[idx][0], idy%3, idz); 
        RyT[idx][idz][idy%3] = Ry[idx][idy%3][idz]; 
    }
}

template <typename scalar_t>
__global__ void _Base_Matrix_H_Kernel(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Rx, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> RxT, 
        const unsigned int dim_x)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_x || idy >= 3 || idz  >= 3){return;}
    _rx(RxT[idx][idz][idy], -atan2(Rx[idx][2][0], Rx[idx][1][0]), idy, idz); 
}

template <typename scalar_t>
__global__ void _V0_deltaK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> dNu, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> met_xy, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> shape,  
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H, 
        const unsigned int dim_i)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_i || idy >= 3 || idz >= 3){ return; } 

    dNu[idx][idy][idz] = met_xy[idx][idy][2 - idz] - H[idx][idy][idz]; 
    scalar_t dot_ji = 0; 
    for (unsigned int i(0); i < 3; ++i)
    {
        dot_ji += (met_xy[idx][i][2 - idz] - H[idx][i][idz])*shape[idx][idy][i]; 
    }
    X[idx][idz][idy] = dot_ji; 
} 

template <typename scalar_t>
__global__ void _DerivativeK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X, 
        const unsigned int dim_i)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if ( idx >= dim_i || idy >= 3 || idz >= 3 ){ return; }
    out[idx][idy][idz] = 0; 
    if (idy == 2 || idz == 2){ return; }
    _pihalf(out[idx][idy][idz]);  
    _rz(out[idx][idy][idz], out[idx][idy][idz], idy, idz); 
}

template <typename scalar_t>
__global__ void _transSumK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> in, 
        const unsigned int dim_i)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if ( idx >= dim_i || idy >= 3 || idz >= 3 ){ return; }
    out[idx][idy][idz] = in[idx][idy][idz] + in[idx][idz][idy]; 
}

template <typename scalar_t>
__global__ void _SwapAB_K(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> DetA, 
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> DetB,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> B,
        const unsigned int dim_i)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_i || idy >= 3 || idz >= 3){ return; }
    _swapAB(A[idx][idy][idz], B[idx][idy][idz], DetA[idx][0], DetB[idx][0]); 
} 

template <typename scalar_t>
__global__ void _imagineK(
        torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> G,
        const torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> A,
        const torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> B,
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> eigs, 
        const unsigned int dim_eig, const unsigned int dim_i)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y%3; 
    const unsigned int idz = blockIdx.y/3; 
    const unsigned int id_eig = blockIdx.z; 
    if ( idx >= dim_i || idy >= 3 || idz >= 3 || id_eig >= dim_eig ){ return; }
    const c10::complex<double> cmplx = eigs[idx][id_eig];
    _imageG(G[idx][id_eig][idy][idz], A[idx][idy][idz], B[idx][idy][idz], cmplx.real(), cmplx.imag()); 
}

template <typename scalar_t>
__global__ void _degenerateK(
        torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> L,
        torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> swps,
        const torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> G,
        const unsigned int dim_eig, const unsigned int dim_i, 
        unsigned int* sy, unsigned int* sz)
{

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y%3; 
    const unsigned int idz = blockIdx.y/3; 
    const unsigned int id_eig = blockIdx.z; 

    if ( idx >= dim_i || blockIdx.y >= 9 || id_eig >= dim_eig ){ return; }

    const bool _dg = G[idx][id_eig][0][0] == 0 && G[idx][id_eig][1][1] == 0; 
    if (_dg && idy == 2){ return; }
    if (_dg && idy == 0)
    {
        if (idz == 0){ L[idx][id_eig][idy][idz] = G[idx][id_eig][0][1]; }
        else if (idz == 1){ L[idx][id_eig][idy][idz] = 0; }
        else if (idz == 2){ L[idx][id_eig][idy][idz] = G[idx][id_eig][1][2]; }
        return; 
    }
    
    if (_dg && idy == 1)
    {
        if (idz == 0){ L[idx][id_eig][idy][idz] = 0; }
        else if (idz == 1){ L[idx][id_eig][idy][idz] = G[idx][id_eig][0][1]; }
        else if (idz == 2){ L[idx][id_eig][idy][idz] = G[idx][id_eig][0][2] - G[idx][id_eig][1][2]; }
        return; 
    }

    bool _swp = abs(G[idx][id_eig][0][0]) > abs(G[idx][id_eig][1][1]); 
    if (idy == 0 && idz == 0){ swps[idx][id_eig] = _swp; }
    const unsigned int _o = (_swp) ? 0 : 9; 
    double _l   = G[idx][id_eig][sy[blockIdx.y + _o]][sz[blockIdx.y + _o]]; 
    double _l11 = G[idx][id_eig][sy[_o + 4]][sz[_o + 4]]; 
    L[idx][id_eig][idy][idz] =  _l / _l11; 
    
}

template <typename scalar_t>
__global__ void _CoFactorK(
        torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> G, 
        torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> L, 
        const unsigned int dim_eig, const unsigned int dim_i,
        unsigned int* dy, unsigned int* dz)
{

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = (blockIdx.y%3)*4; 
    const unsigned int idz = (blockIdx.y/3)*4; 
    const unsigned int id_eig = blockIdx.z; 
    if ( idx >= dim_i || blockIdx.y >= 9 || id_eig >= dim_eig ){ return; }
    
    G[idx][id_eig][blockIdx.y%3][blockIdx.y/3] = _det(
            L[idx][id_eig][dy[idy  ]][dz[idz  ]], L[idx][id_eig][dy[idy+1]][dz[idz+1]], 
            L[idx][id_eig][dy[idy+2]][dz[idz+2]], L[idx][id_eig][dy[idy+3]][dz[idz+3]]);
    if ((blockIdx.y%3+blockIdx.y/3)%2 == 1){ G[idx][id_eig][blockIdx.y%3][blockIdx.y/3] *= -1; }
}

template <typename scalar_t>
__global__ void _FactorizeK(
        torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> O, 
        torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> Q, 
        torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> QCoef, 
        const unsigned int dim_eig, const unsigned int dim_i, const double null)
{

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = (blockIdx.y%3); 
    const unsigned int idz = (blockIdx.y/3); 
    const unsigned int id_eig = blockIdx.z; 
    if ( idx >= dim_i || blockIdx.y >= 9 || id_eig >= dim_eig ){ return; }

    double cq22 = QCoef[idx][id_eig][2][2];
    double q01 = Q[idx][id_eig][0][1]; 
    double q11 = Q[idx][id_eig][1][1]; 

    if ( -cq22 <= null)
    {
        double cq00 = QCoef[idx][id_eig][0][0]; 
        if (-cq00 < 0 ){ return; } 
        
        const unsigned int l = (cq00 == 0) ? 0 : 1; 
        if (idy <= l && idz == 0){ O[idx][id_eig][idy][idz] = q01; return; }
        if (idy <= l && idz == 1){ O[idx][id_eig][idy][idz] = q11; return; }
        if (idy <= l && idz == 2)
        { 
            O[idx][id_eig][idy][idz] = _qsub(Q[idx][id_eig][1][2], cq00, idy);
            return; 
        }
        return; 
    } 
    if ( -cq22 < 0 ){ return; }
    const unsigned int l = (cq22 == 0) ? 0 : 1;  
    if (idy <= l && idz == 0)
    {
        O[idx][id_eig][idy][idz] = _qsub(q01, cq22, idy); 
        return; 
    }

    if (idy <= l && idz == 1)
    {
        O[idx][id_eig][idy][idz] = q11; 
        return; 
    }

    if (idy <= l && idz == 2)
    {
        double cq02 = QCoef[idx][id_eig][0][2]; 
        double cq12 = QCoef[idx][id_eig][1][2]; 
        O[idx][id_eig][idy][idz] = _qsub(q01, q11, cq02, cq12, cq22, idy); 
        return; 
    }
}

