#include <torch/torch.h>
#include "operators.cu"

template <typename scalar_t>
__global__ void _DotK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> i, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> j, 
        const unsigned int dim_i, 
        const unsigned int dim_j)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    if (idx >= dim_i || idy >= dim_j){return;}
    dot_ij(i[idx][idy], j[idx][idy]); 
}

template <typename scalar_t>
__global__ void _DotK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out,
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v1, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v2, 
        const unsigned int dim_z, 
        const unsigned int dim_i1,
        const unsigned int dim_co, 
        const unsigned int grid)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idz >= dim_z || idy >= dim_co || idx >= grid){return;}
    const unsigned int id = idx/dim_i1;
    const unsigned int idx_ = idx%dim_i1; 
    dot_ij(out[idz][idx_][idy + id*dim_co], v1[idz][idx_][idy], v2[idz][idy][id]); 
}

template <typename scalar_t>
__global__ void _DotK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> v1, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> v2, 
        unsigned int dim_x, 
        unsigned int dim_y)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_x || idy >= dim_y || idz >= 3){return;}
    if (idz == 0){ dot_ij(out[idx][idy][idz], v1[idx][idy], v1[idx][idy]); return; }
    if (idz == 1){ dot_ij(out[idx][idy][idz], v2[idx][idy], v2[idx][idy]); return; }
    dot_ij(out[idx][idy][idz], v1[idx][idy], v2[idx][idy]); 
}

template <typename scalar_t>
__global__ void _SumK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const unsigned int length, 
        const unsigned int len_j)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= length){ return; }
    for (unsigned int i(1); i < len_j; ++i)
    {
        sum(pmc[idx][0], pmc[idx][i]);  
    }
}

template <typename scalar_t>
__global__ void _SumK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> tmp, 
        const unsigned int length, const unsigned int len_j)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    if (idx >= length || idy >= 3){ return; }
    for (unsigned int i(0); i < len_j; ++i)
    {
        sum(out[idx][idy], tmp[idx][i][idy]);  
        tmp[idx][i][idy] = 0; 
    }
}

template <typename scalar_t>
__global__ void _SumK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> mul, 
        const unsigned int dim_z,
        const unsigned int dim_x,  
        const unsigned int dim_y, 
        const unsigned int range)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idz >= dim_z || idy >= dim_y || idx >= dim_x){return;}
    
    for (unsigned int i(0); i < range; ++i)
    {
        sum(out[idz][idx][idy], mul[idz][idx][range*idy+i]);  
    }
}

template <typename scalar_t>
__global__ void _CosThetaK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> inpt, 
        const unsigned int length)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= length){ return; }    
    costheta(inpt[idx][0], inpt[idx][0], inpt[idx][1], inpt[idx][2]); 
}

template <typename scalar_t>
__global__ void _SinThetaK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> inpt, 
        const unsigned int length)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= length){ return; }    
    sintheta(inpt[idx][0], inpt[idx][0], inpt[idx][1], inpt[idx][2]); 
}

template <typename scalar_t>
__global__ void _RotK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
        const unsigned int dim_x, const unsigned int dim_r)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_x || idy >= 3 || idz  >= 3){return;}
    if (dim_r == 0){_rx(out[idx][idy][idz], angle[idx][0], idy, idz); return;} 
    if (dim_r == 1){_ry(out[idx][idy][idz], angle[idx][0], idy, idz); return;}    
    if (dim_r == 2){_rz(out[idx][idy][idz], angle[idx][0], idy, idz); return;} 
}

template <typename scalar_t>
__global__ void _CoFactorK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> mtx, 
        const unsigned int dim_x, const unsigned int dim_y)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= dim_x || idy >= dim_y || idz  >= 3){return;}
    const unsigned int _y[12] = {1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1}; 
    const unsigned int _z[12] = {1, 2, 1, 2, 0, 2, 0, 2, 0, 1, 0, 1}; 
    const unsigned int idy_ = idy*4; 
    const unsigned int idz_ = idz*4; 
    _det(
            out[idx][idy][idz], 
            mtx[idx][_y[idy_  ]][_z[idz_  ]], 
            mtx[idx][_y[idy_+1]][_z[idz_+1]], 
            mtx[idx][_y[idy_+2]][_z[idz_+2]], 
            mtx[idx][_y[idy_+3]][_z[idz_+3]]
    ); 
    if ((idy+idz)%2 == 1){ out[idx][idy][idz] *= -1; }
}

template <typename scalar_t>
__global__ void _DetDotK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> coeff, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix, 
        const unsigned int len)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= len || idy >= 1 || idz >= 3){ return; } 
    out[idx][idy][idz]  = coeff[idx][idy][idz]*matrix[idx][idy][idz]; 
}

template <typename scalar_t>
__global__ void _DetSumK(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> det, 
        const unsigned int len)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= len){ return; } 
    out[idx][0] = det[idx][0][0] + det[idx][0][1] + det[idx][0][2]; 
}

template <typename scalar_t>
__global__ void _InvK(
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> det, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> coef, 
        const unsigned int len)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y; 
    const unsigned int idz = blockIdx.z; 
    if (idx >= len || idy >= 3 || idz >= 3){ return; } 
    if (det[idx][0] == 0){ out[idx][idz][idy] = 0; return; }
    out[idx][idz][idy] = (1/det[idx][0]) * coef[idx][idy][idz]; 
}


