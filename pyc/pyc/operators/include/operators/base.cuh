#ifndef CU_OPERATORS_BASE_H
#define CU_OPERATORS_BASE_H
#include <atomic/cuatomic.cuh>

template <typename scalar_t>
__global__ void _dot(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> v1, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> v2, 
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> out, 
        const unsigned int dx, const unsigned int dy
){

    extern __shared__ double sdata[]; 
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = (blockIdx.y * blockDim.y + threadIdx.y); 
    const unsigned int _idz = (_idx*dy + _idy)%dy; 

    if (_idx >= dx || _idy >= dy){return;}
    sdata[_idz] = v1[_idx][_idy] * v2[_idx][_idy];  
    __syncthreads(); 
    if (_idz){return;}
    for (size_t x(1); x < dy; ++x){sdata[0] += sdata[x];}
    out[_idx] = sdata[0]; 
}

template <typename scalar_t>
__global__ void _costheta(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x,
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out,
        const unsigned int dx, const unsigned int dy, bool get_sin = false
){
    extern __shared__ double sdata[]; 
    __shared__ double smem[3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = (_idx*dy + _idy)%dy; 

    if (_idx >= dx || _idy >= dy){return;}
    sdata[_idz*2]   = x[_idx][_idy];
    sdata[_idz*2+1] = y[_idx][_idy]; 
    if (_idz >= 3){return;}
    __syncthreads(); 

    const unsigned int o1x  = (threadIdx.y) ? 1 : 0; 
    const unsigned int o2x  = (threadIdx.y > 1) ? 1 : 0; 

    double sm = 0; 
    for (size_t x(0); x < dy; ++x){sm += sdata[x*2 + o1x] * sdata[x*2 + o2x];} 
    smem[_idz] = sm; 

    if (_idz){return;}
    __syncthreads(); 
    double cs = _cmp(smem[0], smem[2], smem[1]); 
    if (!get_sin){out[_idx][0] = cs; return; }
    cs = 1 - pow(cs, 2); 
    out[_idx][0] = _sqrt(&cs); 
}

template <typename scalar_t>
__global__ void _rx(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const unsigned int dx
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
  
    scalar_t vl = angle[_idx][0];  
    out[_idx][_idy][_idz] = _rx(&vl, _idy, _idz); 
}



template <typename scalar_t>
__global__ void _ry(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const unsigned int dx
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
  
    scalar_t vl = angle[_idx][0];  
    out[_idx][_idy][_idz] = _ry(&vl, _idy, _idz); 
}


template <typename scalar_t>
__global__ void _rz(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> angle, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const unsigned int dx
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
  
    scalar_t vl = angle[_idx][0];  
    out[_idx][_idy][_idz] = _rz(&vl, _idy, _idz); 
}


template <typename scalar_t>
__global__ void _rt(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc, 
        const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> phi, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> theta, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out
){
    __shared__ double pmx[3]; 
    __shared__ double pmr[9][3];
    __shared__ double pmd[9][3];

    __shared__ double rz[3][3];
    __shared__ double ry[3][3];

    __shared__ double rxt[3][3]; 
    __shared__ double rzt[3][3];
    __shared__ double ryt[3][3];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _blk = (blockIdx.y * blockDim.y + threadIdx.y)*3 + blockIdx.z * blockDim.z + threadIdx.z; 
    const unsigned int _idy  = _blk/3;
    const unsigned int _idz  = _blk%3; 

    double phi_   = -phi[_idx]; 
    double theta_ = 0.5*M_PI - theta[_idx][0]; 
    pmx[_idz] = pmc[_idx][_idz]; 

    double rz_ = _rz(&phi_  , _idy, _idz); 
    double ry_ = _ry(&theta_, _idy, _idz); 

    rz[_idy][_idz]  = rz_; 
    ry[_idy][_idz]  = ry_;  

    rzt[_idz][_idy] = rz_; 
    ryt[_idz][_idy] = ry_;  
    __syncthreads(); 

    for (size_t x(0); x < 3; ++x){
        double sm = 0; 
        for (size_t y(0); y < 3; ++y){sm += pmx[y] * rz[x][y];}
        pmr[_blk][x] = sm; 
    }
    for (size_t x(0); x < 3; ++x){
        double sm = 0; 
        for (size_t y(0); y < 3; ++y){sm += pmr[_blk][y] * ry[x][y];}
        pmd[_blk][x] = sm; 
    }

    double smz = -atan2(pmd[_blk][2], pmd[_blk][1]); 
    rxt[_idz][_idy] = _rx(&smz, _idy, _idz); 
    __syncthreads(); 

    double sx = 0; 
    for (size_t x(0); x < 3; ++x){sx += ryt[_idz][x] * rxt[x][_idy];}
    pmr[_idz][_idy] = sx; 
    __syncthreads(); 

    double sy = 0; 
    for (size_t x(0); x < 3; ++x){sy += rzt[_idz][x] * pmr[x][_idy];}
    out[_idx][_idz][_idy] = sy; 
}


__device__ __constant__ const unsigned int _x[12] = {1, 1, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1}; 
__device__ __constant__ const unsigned int _y[12] = {1, 2, 1, 2, 0, 2, 0, 2, 0, 1, 0, 1}; 

template <typename scalar_t>
__global__ void _cofactor(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out
){
    __shared__ double mat[3][3]; 
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 

    const unsigned int idy = threadIdx.y*4;
    const unsigned int idz = threadIdx.z*4;  

    mat[threadIdx.y][threadIdx.z] = matrix[_idx][threadIdx.y][threadIdx.z]; 
    __syncthreads();

    double ad = mat[ _x[idy  ] ][ _y[idz  ] ] * mat[ _x[idy+3] ][ _y[idz+3] ]; 
    double bc = mat[ _x[idy+1] ][ _y[idz+1] ] * mat[ _x[idy+2] ][ _y[idz+2] ]; 
    double cf = pow(-1, int(threadIdx.y) + int(threadIdx.z)); 
    out[_idx][threadIdx.y][threadIdx.z] = (ad - bc)*cf;
}

template <typename scalar_t>
__global__ void _determinant(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out
){
    __shared__ double mat[3][3]; 
    __shared__ double det[3][3];
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = threadIdx.y*4;
    const unsigned int idz = threadIdx.z*4;  

    mat[threadIdx.y][threadIdx.z] = matrix[_idx][threadIdx.y][threadIdx.z]; 
    __syncthreads();

    double ad = mat[ _x[idy  ] ][ _y[idz  ] ] * mat[ _x[idy+3] ][ _y[idz+3] ]; 
    double bc = mat[ _x[idy+1] ][ _y[idz+1] ] * mat[ _x[idy+2] ][ _y[idz+2] ]; 
    double cf = pow(-1, int(threadIdx.y) + int(threadIdx.z)); 
    det[threadIdx.y][threadIdx.z] = (ad - bc)*mat[threadIdx.y][threadIdx.z]*cf;
    if (threadIdx.y || threadIdx.z){return;}
    __syncthreads(); 
    out[_idx][0] = det[0][0] + det[0][1] + det[0][2];  
}

template <typename scalar_t>
__global__ void _inverse(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> inv,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> det
){
    __shared__ double _mat[3][3];
    __shared__ double _cof[3][3];  
    __shared__ double _det[3][3];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = threadIdx.y*4;
    const unsigned int idz = threadIdx.z*4;  

    _mat[threadIdx.y][threadIdx.z] = matrix[_idx][threadIdx.y][threadIdx.z]; 
    double cf = pow(-1, int(threadIdx.y) + int(threadIdx.z)); 
    __syncthreads();

    double ad = _mat[ _x[idy  ] ][ _y[idz  ] ] * _mat[ _x[idy+3] ][ _y[idz+3] ]; 
    double bc = _mat[ _x[idy+1] ][ _y[idz+1] ] * _mat[ _x[idy+2] ][ _y[idz+2] ]; 
    double mx = (ad - bc)*cf; 

    _cof[threadIdx.z][threadIdx.y] = mx;  // transpose cofactor matrix to get adjoint 
    _det[threadIdx.y][threadIdx.z] = mx * _mat[threadIdx.y][threadIdx.z]; 
    __syncthreads(); 

    double _dt = _det[0][0] + _det[0][1] + _det[0][2];
 
    if (!threadIdx.y && !threadIdx.z){det[_idx][0] = _dt;}
    inv[_idx][threadIdx.y][threadIdx.z] = _cof[threadIdx.y][threadIdx.z]*_div(&_dt); 
}





#endif
