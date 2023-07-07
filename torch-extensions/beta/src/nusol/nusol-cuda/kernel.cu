#include <torch/torch.h>
#include "nusol.cu"

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
        const unsigned int dim_i)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_i || idy >= 3 || idz >= 3){ return; }    
    if (idy <= 1 && idz == 1){ return; }
    if (idy == 2 && (idz == 0 || idz == 2)){ return; }

    scalar_t beta_mu   = sqrt(beta2_mu[idx][0]);
    scalar_t beta_b    = sqrt(beta2_b[idx][0]); 
    scalar_t mass2_W   = mass2[idx][0]*mass2[idx][0];  
    scalar_t mass2_top = mass2[idx][1]*mass2[idx][1];  
    scalar_t mass2_nu  = mass2[idx][2]*mass2[idx][2];
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
__global__ void _Nu_deltaK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> met_xy, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> sig,  
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H, 
        const unsigned int dim_i, const unsigned int sig_i)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_i || idy >= 3 || idz >= 3){ return; } 
    scalar_t sig_    = 0; 
    scalar_t _met_yz = 0; 
    scalar_t _met_zy = 0; 
    if (idx < sig_i){ sig_ = sig[idx][idz][idy]; }
    else {sig_ = sig[0][idz][idy]; }
    if (idy == 2 && idz == 2){ sig_ = 0; }

    if (idy <= 1 && idz == 2){ _met_yz = met_xy[idx][idy]; }
    if (idz <= 1 && idy == 2){ _met_zy = met_xy[idx][idz]; }
    _met_yz = _met_yz - H[idx][idy][idz]; 
    // need to fix this!
    X[idx][idy][idz] = (_met_zy - H[idx][idz][idy])*sig_; //*_met_yz; 
} 

