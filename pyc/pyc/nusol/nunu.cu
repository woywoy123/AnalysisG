#include <nusol/base.cuh>
#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <operators/operators.cuh>
#include <transform/transform.cuh>

template <typename scalar_t, size_t size_x>
__global__ void _nunu_init_(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> met_xy,
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H1_inv, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H2_inv,
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H1, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H2,

              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n_,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> N,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K_,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> S
){

    __shared__ double _H1[size_x][3][3];
    __shared__ double _H1inv[size_x][3][3];
    __shared__ double _H1invT[size_x][3][3]; 

    __shared__ double _H2[size_x][3][3];
    __shared__ double _H2inv[size_x][3][3]; 
    __shared__ double _H2invT[size_x][3][3]; 

    __shared__ double _N1[size_x][3][3]; 
    __shared__ double _N2[size_x][3][3]; 

    __shared__ double _uC[size_x][3][3]; 
    __shared__ double _V0[size_x][3][3];

    __shared__ double _S_[size_x][3][3];
    __shared__ double _ST[size_x][3][3];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idx = threadIdx.x; 
    const unsigned int idy = threadIdx.y; 
    const unsigned int idz = threadIdx.z; 
    if (_idx >= S.size({0})){return;}

    _H1[idx][idy][idz] = H1[_idx][idy][idz]; 
    _H2[idx][idy][idz] = H2[_idx][idy][idz]; 

    _H1inv[idx][idy][idz]  = H1_inv[_idx][idy][idz]; 
    _H2inv[idx][idy][idz]  = H2_inv[_idx][idy][idz]; 

    _H1invT[idx][idz][idy] = _H1inv[idx][idy][idz]; 
    _H2invT[idx][idz][idy] = _H2inv[idx][idy][idz]; 
    
    _V0[idx][idy][idz] = met_xy[_idx][idy] * (idz == 2 && idy < 2);
    _uC[idx][idy][idz] = (idy == idz)*(2*(idy < 2) - 1);
    __syncthreads();

    _N1[idx][idy][idz] = _dot(_H1invT[idx], _uC[idx], idy, idz, 3); 
    _N2[idx][idy][idz] = _dot(_H2invT[idx], _uC[idx], idy, idz, 3); 

    _S_[idx][idy][idz] = _V0[idx][idy][idz] - _uC[idx][idy][idz]; 
    _ST[idx][idz][idy] = _V0[idx][idy][idz] - _uC[idx][idy][idz];
    __syncthreads(); 

    _H1invT[idx][idy][idz] = _dot(_N1[idx], _H1inv[idx], idy, idz, 3); 
    _H2invT[idx][idy][idz] = _dot(_N2[idx], _H2inv[idx], idy, idz, 3); 
    __syncthreads(); 
    
    _N1[idx][idy][idz] = _H1invT[idx][idy][idz]; 
    _N2[idx][idy][idz] = _H2invT[idx][idy][idz]; 
    __syncthreads(); 

    _V0[idx][idy][idz] = _dot(_ST[idx], _N2[idx], idy, idz, 3); 
    __syncthreads();

    N[_idx][idy][idz]  = _N1[idx][idy][idz]; 
    S[_idx][idy][idz]  = _S_[idx][idy][idz]; 
    K[_idx][idy][idz]  = _dot(_H1[idx], _H1inv[idx], idy, idz, 3); 
    K_[_idx][idy][idz] = _dot(_H2[idx], _H2inv[idx], idy, idz, 3); 
    n_[_idx][idy][idz] = _dot(_V0[idx], _S_[idx]   , idy, idz, 3); 
}

template <typename scalar_t>
__global__ void _nunu_vp_(
        const torch::PackedTensorAccessor64<long    , 3, torch::RestrictPtrTraits> srt, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> S, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K_,
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v_,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu1,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu2,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> dst
){
    __shared__ double  _v[18][3]; 
    __shared__ double __v[18][3]; 
    __shared__ double  _S[18][3][3]; 
    __shared__ double  _K[18][3][3];
    __shared__ double __K[18][3][3]; 
    __shared__ double nus[18][2][3]; 
    __shared__ double _sd[18]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    const long idx  = srt[_idx][threadIdx.y][0];
    if (!threadIdx.z){_sd[_idy] = dst[_idx][idx][0];}

    _v[_idy][_idz] = v[_idx][idx][threadIdx.z]; 
    for (size_t x(0); x < 3; ++x){ _S[_idy][_idz][x] =  S[_idx][_idz][x];}
    for (size_t x(0); x < 3; ++x){ _K[_idy][_idz][x] =  K[_idx][_idz][x];}
    for (size_t x(0); x < 3; ++x){__K[_idy][_idz][x] = K_[_idx][_idz][x];}
    __syncthreads();

    __v[_idy][_idz] = _dot(_S[_idy][_idz], _v[_idy], 3); 
    v_[_idx][_idy][_idz] = __v[_idy][_idz]; 
    __syncthreads(); 

    nus[_idy][0][_idz] = _dot( _K[_idy][_idz],  _v[_idy], 3)*(_sd[_idy] < 200); 
    nus[_idy][1][_idz] = _dot(__K[_idy][_idz], __v[_idy], 3)*(_sd[_idy] < 200); 

    nu1[_idx][_idy][_idz] = nus[_idy][0][_idz]; 
    nu2[_idx][_idy][_idz] = nus[_idy][1][_idz]; 
    if (threadIdx.z){return;}
    bool null = _sum(nus[_idy][0], 3) * _sum(nus[_idy][1], 3);
    dst[_idx][_idy][0] = _sd[_idy] * null;
}


std::map<std::string, torch::Tensor> nusol_::NuNu(
        torch::Tensor* H1_, torch::Tensor* H1_inv, 
        torch::Tensor* H2_, torch::Tensor* H2_inv, 
        torch::Tensor* met_xy, double null
){
    const unsigned int dx = H1_ -> size({0}); 
    const unsigned int thx = (dx >= 48) ? 48 : dx; 
    const dim3 thd = dim3(thx, 3, 3);
    const dim3 blk = blk_(dx, thx, 3, 3, 3, 3); 
    
    torch::Tensor S  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor N  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor K  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor n_ = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor K_ = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 

    AT_DISPATCH_ALL_TYPES(H1_ -> scalar_type(), "NuNu", [&]{
        _nunu_init_<scalar_t, 48><<<blk, thd>>>(
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                H1_inv -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                H2_inv -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H1_ -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H2_ -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),

                       n_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        N.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        K.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       K_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        S.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
    }); 

    std::map<std::string, torch::Tensor> out = nusol_::Intersection(&N, &n_, null); 
    torch::Tensor nu1 = torch::zeros_like(out["solutions"]); 
    torch::Tensor nu2 = torch::zeros_like(out["solutions"]); 
    torch::Tensor v_  = torch::zeros_like(out["solutions"]); 
    torch::Tensor v   = out["solutions"]; 
    torch::Tensor ds  = out["distances"]; 
    torch::Tensor srt = std::get<1>(ds.sort(-2, false));

    const dim3 thN = dim3(1, 18, 3);
    const dim3 blN = blk_(dx, 1, 18, 18, 3, 3); 
    AT_DISPATCH_ALL_TYPES(H1_ -> scalar_type(), "NuNu", [&]{
        _nunu_vp_<scalar_t><<<blN, thN>>>(
                      srt.packed_accessor64<long    , 3, torch::RestrictPtrTraits>(),
                        S.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        K.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       K_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        v.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                       v_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                      nu1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                      nu2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       ds.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); 
    }); 

    out["n_"]  = n_; 
    out["nu1"] = nu1; 
    out["nu2"] = nu2; 
    return out;  
}

std::map<std::string, torch::Tensor> nusol_::NuNu(
            torch::Tensor* pmc_b1,  torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy,  double null, torch::Tensor* m1, torch::Tensor* m2
){
    if (!m2){m2 = m1;}
    std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(pmc_b1, pmc_mu1, m1);
    torch::Tensor H1_inv = std::get<0>(operators_::Inverse(&H1_m["H_perp"])); 
    torch::Tensor H1_    = H1_m["H"]; 

    std::map<std::string, torch::Tensor> H2_m = nusol_::BaseMatrix(pmc_b2, pmc_mu2, m2);
    torch::Tensor H2_inv = std::get<0>(operators_::Inverse(&H2_m["H_perp"])); 
    torch::Tensor H2_    = H2_m["H"]; 

    torch::Tensor passed = H1_m["passed"] * H2_m["passed"]; 
    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&H1_, &H1_inv, &H2_, &H2_inv, met_xy, null); 
    out["passed"] = passed; 
    return out; 
}

std::map<std::string, torch::Tensor> nusol_::NuNu(
            torch::Tensor* pmc_b1,  torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy, double null, double massT1, double massW1, double massT2, double massW2
){
    std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(pmc_b1, pmc_mu1, massT1, massW1, 0);
    torch::Tensor H1_inv = std::get<0>(operators_::Inverse(&H1_m["H_perp"])); 
    torch::Tensor H1_    = H1_m["H"]; 

    std::map<std::string, torch::Tensor> H2_m = nusol_::BaseMatrix(pmc_b2, pmc_mu2, massT2, massW2, 0);
    torch::Tensor H2_inv = std::get<0>(operators_::Inverse(&H2_m["H_perp"])); 
    torch::Tensor H2_    = H2_m["H"]; 
    torch::Tensor passed = H1_m["passed"] * H2_m["passed"]; 
    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&H1_, &H1_inv, &H2_, &H2_inv, met_xy, null); 
    out["passed"] = passed; 
    return out; 
}




