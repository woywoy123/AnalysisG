#include <nusol/base.cuh>
#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <operators/operators.cuh>
#include <transform/transform.cuh>

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





std::map<std::string, torch::Tensor> nusol_::Nu(
        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
        torch::Tensor* masses, torch::Tensor* sigma , double null
){
    torch::Tensor H = nusol_::BaseMatrix(pmc_b, pmc_mu, masses)["H"];
    torch::Tensor X = torch::zeros_like(H); 
    torch::Tensor M = torch::zeros_like(H); 
    torch::Tensor Unit = torch::zeros_like(H); 
   
    const unsigned int dx = pmc_b -> size({0}); 
    const dim3 thd = dim3(1, 3, 3);
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "Nu", [&]{
        _nu_init_<scalar_t><<<blk, thd>>>(
                 sigma -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        M.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        Unit.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());

    }); 

    std::map<std::string, torch::Tensor> out = nusol_::Intersection(&M, &Unit, null); 

    const dim3 thN = dim3(1, 18, 3);
    const dim3 blN = blk_(dx, 1, 18, 18, 3, 3); 
    torch::Tensor nu   = torch::zeros({dx, 18, 3}, MakeOp(pmc_b)); 
    torch::Tensor chi2 = torch::zeros({dx, 18, 1}, MakeOp(pmc_b)); 
    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "Nu", [&]{
        _chi2<scalar_t><<<blN, thN>>>(
         out["solutions"].packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
         out["distances"].packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       nu.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     chi2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
    }); 

    out["X"] = X; 
    out["M"] = M; 
    out["nu"] = nu; 
    out["chi2"] = chi2;
    return out; 
}


std::map<std::string, torch::Tensor> nusol_::Nu(
        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
        torch::Tensor* sigma, double null, double massT, double massW
){
    torch::Tensor H = nusol_::BaseMatrix(pmc_b, pmc_mu, massT, massW, 0)["H"];
    torch::Tensor X = torch::zeros_like(H); 
    torch::Tensor M = torch::zeros_like(H); 
    torch::Tensor Unit = torch::zeros_like(H); 
   
    const unsigned int dx = pmc_b -> size({0}); 
    const dim3 thd = dim3(1, 3, 3);
    const dim3 blk = blk_(dx, 1, 3, 3, 3, 3); 
    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "Nu", [&]{
        _nu_init_<scalar_t><<<blk, thd>>>(
                 sigma -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                        H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        M.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     Unit.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());

    }); 

    std::map<std::string, torch::Tensor> out = nusol_::Intersection(&M, &Unit, null); 

    const dim3 thN = dim3(1, 18, 3);
    const dim3 blN = blk_(dx, 1, 18, 18, 3, 3); 
    torch::Tensor nu   = torch::zeros({dx, 18, 3}, MakeOp(pmc_b)); 
    torch::Tensor chi2 = torch::zeros({dx, 18, 1}, MakeOp(pmc_b)); 
    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "Nu", [&]{
        _chi2<scalar_t><<<blN, thN>>>(
         out["solutions"].packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
         out["distances"].packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       nu.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     chi2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
    }); 

    out["X"] = X; 
    out["M"] = M; 
    out["nu"] = nu; 
    out["chi2"] = chi2;
    return out; 
}


