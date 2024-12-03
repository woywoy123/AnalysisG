#include <nusol/base.cuh>
#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <operators/operators.cuh>
#include <transform/transform.cuh>

template <typename scalar_t, size_t size_x>
__global__ void _nu_init_(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> s2,
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> met_xy,
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H, 

        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> X, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> M, 
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Unit
){

    __shared__ double _H[size_x][3][3];
    __shared__ double _S2[size_x][3][3]; 
    __shared__ double _V0[size_x][3][3]; 

    __shared__ double _dNu[size_x][3][3]; 
    __shared__ double _dNuT[size_x][3][3]; 

    __shared__ double _X[size_x][3][3]; 
    __shared__ double _T[size_x][3][3]; 
    __shared__ double _Dx[size_x][3][3]; 

    __shared__ double _XD[size_x][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idx = threadIdx.x;
    const unsigned int idy = threadIdx.y;
    const unsigned int idz = threadIdx.z;  
    if (_idx >= met_xy.size({0})){return;}

    // ------- Populate data ----------- //
    _S2[idx][idy][idz] = 0; 
    _V0[idx][idy][idz] = 0; 
    _H[idx][idy][idz] = H[_idx][idy][idz]; 

    double pi = M_PI*0.5; 
    _Dx[idx][idy][idz] = _rz(&pi, idy, idz); 
    _T[idx][idy][idz] = (idy == idz)*(idy < 2); 

    if (idy < 2  && idz < 2){_S2[idx][idy][idz] = s2[_idx][idy][idz];}
    if (idz == 2 && idy < 2){_V0[idx][idy][idz] = met_xy[_idx][idy];}
    __syncthreads(); 

    // ------- matrix inversion for S2 ------ //
    if (!idy && !idz){
        double s00 = _S2[idx][0][0]; 
        double s11 = _S2[idx][1][1]; 
        double s01 = _S2[idx][0][1]; 
        double s10 = _S2[idx][1][0]; 
        double det = (s00*s11 - s01*s10);
        det = _div(&det);
    
        // S2^-1 with transpose
        _S2[idx][0][0] =  s11*det; 
        _S2[idx][1][1] =  s00*det; 

        _S2[idx][0][1] = -s10*det; 
        _S2[idx][1][0] = -s01*det; 
    }

    double di = _dot(_Dx[idx], _T[idx], idy, idz, 3); 
    _dNu[idx][idy][idz]  = _V0[idx][idy][idz] - _H[idx][idy][idz]; 
    _dNuT[idx][idz][idy] = _V0[idx][idy][idz] - _H[idx][idy][idz]; 
    __syncthreads(); 

    _Dx[idx][idy][idz] = di; 
    _T[idx][idy][idz] = _dot(_dNuT[idx], _S2[idx], idy, idz, 3); 
    __syncthreads(); 

    _X[idx][idy][idz] = _dot(_T[idx], _dNu[idx], idy, idz, 3); 
    __syncthreads(); 

    _T[idx][idy][idz]  = (idy == idz)*(2*(idy < 2) - 1); 
    _XD[idx][idy][idz] = _dot(_X[idx], _Dx[idx], idy, idz, 3) + _dot(_X[idx], _Dx[idx], idz, idy, 3);  

    X[_idx][idy][idz] = _X[idx][idy][idz]; 
    M[_idx][idy][idz] = _XD[idx][idy][idz]; 
    Unit[_idx][idy][idz] = _T[idx][idy][idz];
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
        torch::Tensor* H, torch::Tensor* sigma, torch::Tensor* met_xy, double null
){
    torch::Tensor X = torch::zeros_like(*H); 
    torch::Tensor M = torch::zeros_like(*H); 
    torch::Tensor Unit = torch::zeros_like(*H); 
   
    const unsigned int dx = H -> size({0}); 
    const unsigned int thx = (dx >= 64) ? 64 : dx; 
    const dim3 thd = dim3(thx, 3, 3);
    const dim3 blk = blk_(dx, thx, 3, 3, 3, 3); 
    AT_DISPATCH_ALL_TYPES(H -> scalar_type(), "Nu", [&]{
        _nu_init_<scalar_t, 64><<<blk, thd>>>(
                 sigma -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     H -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        M.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        Unit.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());

    }); 

    std::map<std::string, torch::Tensor> out = nusol_::Intersection(&M, &Unit, null); 

    const dim3 thN = dim3(1, 18, 3);
    const dim3 blN = blk_(dx, 1, 18, 18, 3, 3); 
    torch::Tensor nu   = torch::zeros({dx, 18, 3}, MakeOp(H)); 
    torch::Tensor chi2 = torch::zeros({dx, 18, 1}, MakeOp(H)); 
    AT_DISPATCH_ALL_TYPES(H -> scalar_type(), "Nu", [&]{
        _chi2<scalar_t><<<blN, thN>>>(
         out["solutions"].packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
         out["distances"].packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     H -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       nu.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                     chi2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
    }); 
    out["X"] = X; 
    out["M"] = M; 
    out["nu"] = nu; 
    out["distances"] = chi2;
    return out; 
}
 
std::map<std::string, torch::Tensor> nusol_::Nu(
        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
        torch::Tensor* masses, torch::Tensor* sigma , double null
){
    torch::Tensor H = nusol_::BaseMatrix(pmc_b, pmc_mu, masses)["H"];
    return nusol_::Nu(&H, sigma, met_xy, null); 
}


std::map<std::string, torch::Tensor> nusol_::Nu(
        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
        torch::Tensor* sigma, double null, double massT, double massW
){
    torch::Tensor H = nusol_::BaseMatrix(pmc_b, pmc_mu, massT, massW, 0)["H"];
    return nusol_::Nu(&H, sigma, met_xy, null); 
}


