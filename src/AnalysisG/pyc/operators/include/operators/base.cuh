#ifndef CU_OPERATORS_BASE_H
#define CU_OPERATORS_BASE_H
#include <atomic/cuatomic.cuh>

template <typename scalar_t>
__global__ void _dot(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v1, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v2, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out, 
        const unsigned int dx, const unsigned int dy
){
    extern __shared__ double sdata[]; 

    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x); 
    const unsigned int _idy = threadIdx.y; 
    const unsigned int _idz = threadIdx.z; 
    const unsigned int idx = threadIdx.x*dy*dy*2; 

    const unsigned int _xi = (_idy * dy + _idz) + idx; 
    const unsigned int _ri = (_xi / dy) % dy;
    const unsigned int _rj = _xi % dy;  
    if (_idx >= dx){return;}

    sdata[_xi] = v1[_idx][_ri][_rj]; 
    sdata[_xi + dy*dy] = v2[_idx][_rj][_ri]; 
    __syncthreads(); 

    double sm = 0; 
    unsigned int rol = _ri*dy + idx; 
    unsigned int col = _rj*dy + dy*dy + idx; 
    for (size_t x(0); x < dy; ++x){sm += sdata[rol + x] * sdata[col + x];}
    out[_idx][_ri][_rj] = sm; 
}


template <typename scalar_t>
__global__ void _cross(
        const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> v1, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v2, 
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> out, 
        const unsigned int dy, const unsigned int dz
){

    __shared__ double Av[9][3][3]; 
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _dtt = threadIdx.y; 
    const unsigned int _idy1 = (_dtt/9); 
    const unsigned int _idy2 = (_dtt%3); 
    const unsigned int _idy3 = (_dtt/3); 
    double v1_ = v1[_idx][_idy1][_idy3][threadIdx.z]; 
    double v2_ = v2[_idx][_idy2][threadIdx.z];
    Av[threadIdx.y][0][threadIdx.z] = v1_; 
    Av[threadIdx.y][1][threadIdx.z] = v1_; 
    Av[threadIdx.y][2][threadIdx.z] = v2_; 
    __syncthreads(); 
    out[_idx][_idy3][_idy2][threadIdx.z] = _cofactor(Av[threadIdx.y], 0, threadIdx.z); 
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
    cs = 1 - cs*cs; 
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
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> phi, 
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

    double phi_   = -phi[_idx][0]; 
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
    pmr[_idz][_idy] = _dot(ryt, rxt, _idz, _idy, 3); 
    __syncthreads(); 
    out[_idx][_idz][_idy] = _dot(rzt, pmr, _idz, _idy, 3); 
}


template <typename scalar_t, size_t size_x>
__global__ void _cofactor(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> out
){
    __shared__ double mat[size_x][3][3]; 
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (_idx >= matrix.size({0})){return;}

    mat[threadIdx.x][threadIdx.y][threadIdx.z] = matrix[_idx][threadIdx.y][threadIdx.z]; 
    __syncthreads();
    out[_idx][threadIdx.y][threadIdx.z] = _cofactor(mat[threadIdx.x], threadIdx.y, threadIdx.z);
}

template <typename scalar_t, size_t size_x>
__global__ void _determinant(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out
){
    __shared__ double mat[size_x][3][3]; 
    __shared__ double det[size_x][3][3];
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (_idx >= matrix.size({0})){return;}
    mat[threadIdx.x][threadIdx.y][threadIdx.z] = matrix[_idx][threadIdx.y][threadIdx.z]; 
    __syncthreads();

    double minor = _cofactor(mat[threadIdx.x], threadIdx.y, threadIdx.z); 
    det[threadIdx.x][threadIdx.y][threadIdx.z] = minor * mat[threadIdx.x][threadIdx.y][threadIdx.z];

    if (threadIdx.y || threadIdx.z){return;}
    __syncthreads(); 
    out[_idx][0] = det[threadIdx.x][0][0] + det[threadIdx.x][0][1] + det[threadIdx.x][0][2];  
}

template <typename scalar_t, size_t size_x>
__global__ void _inverse(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> inv,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> det
){
    __shared__ double _mat[size_x][3][3];
    __shared__ double _cof[size_x][3][3];  
    __shared__ double _det[size_x][3][3];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (_idx >= matrix.size({0})){return;}
    _mat[threadIdx.x][threadIdx.y][threadIdx.z] = matrix[_idx][threadIdx.y][threadIdx.z]; 
    __syncthreads();

    double mx = _cofactor(_mat[threadIdx.x], threadIdx.y, threadIdx.z);
    _cof[threadIdx.x][threadIdx.z][threadIdx.y] = mx;  // transpose cofactor matrix to get adjoint 
    _det[threadIdx.x][threadIdx.y][threadIdx.z] = mx * _mat[threadIdx.x][threadIdx.y][threadIdx.z]; 
    __syncthreads(); 

    double _dt = _det[threadIdx.x][0][0] + _det[threadIdx.x][0][1] + _det[threadIdx.x][0][2];
    if (!threadIdx.y && !threadIdx.z){det[_idx][0] = _dt;}
    inv[_idx][threadIdx.y][threadIdx.z] = _cof[threadIdx.x][threadIdx.y][threadIdx.z]*_div(&_dt); 
}

template <typename scalar_t, size_t size_x>
__global__ void _eigenvalue(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> matrix,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> real,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> img
){
    
    __shared__ double _mat[size_x][3][3]; 
    __shared__ double _cof[size_x][3][3]; 
    __shared__ double _det[size_x][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idx = threadIdx.x; 
    const unsigned int idy = threadIdx.y;
    const unsigned int idz = threadIdx.z; 

    if (_idx >= matrix.size({0})){return;}
    _mat[idx][idy][idz] = matrix[_idx][idy][idz]; 
    __syncthreads();
    
    double minor = _cofactor(_mat[idx], idy, idz, false); 
    _cof[idx][idy][idz] = -minor;
    _det[idx][idy][idz] = pow(-1, int(idy) + int(idz)) * minor * _mat[idx][idy][idz]; 
    __syncthreads(); 

    c10::complex<double> tr = _mat[idx][0][0] + _mat[idx][1][1] + _mat[idx][2][2]; 
    c10::complex<double> c  = _cof[idx][0][0] + _cof[idx][1][1] + _cof[idx][2][2]; 
    c10::complex<double> d  = _det[idx][0][0] + _det[idx][0][1] + _det[idx][0][2];  
    c10::complex<double> o  = c10::complex<double>( -0.5, 0.5 * sqrt( 3.0 ) ); 
    c10::complex<double> o2 = o * o;

    c10::complex<double> p = (tr * tr + 3.0 * c) / 9.0; 
    c10::complex<double> q = (9.0 * tr * c + 27.0 * d + 2.0 * tr * tr * tr) / 54.0; 
    c10::complex<double> dt = q * q - p * p * p; 

    c10::complex<double> g1 = std::pow(q + std::sqrt(dt), 1.0 / 3.0 );
    c10::complex<double> g2 = std::pow(q - std::sqrt(dt), 1.0 / 3.0 );

    c10::complex<double> offset = tr / 3.0;
    c10::complex<double> lmb1 = g1      + g2      + offset;
    c10::complex<double> lmb2 = g1 * o  + g2 * o2 + offset;
    c10::complex<double> lmb3 = g1 * o2 + g2 * o  + offset;
 
    if (idz == 0 && idy == 0){real[_idx][0] = lmb1.real(); return;}
    if (idz == 0 && idy == 1){img [_idx][0] = _clp(lmb1.imag()); return;}
    if (idz == 0 && idy == 2){real[_idx][1] = lmb2.real(); return;}
    if (idz == 1 && idy == 0){img [_idx][1] = _clp(lmb2.imag()); return;}
    if (idz == 1 && idy == 1){real[_idx][2] = lmb3.real(); return;}
    if (idz == 1 && idy == 2){img [_idx][2] = _clp(lmb3.imag()); return;}
}

#endif
