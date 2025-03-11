#include <nusol/base.cuh>
#include <utils/utils.cuh>
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

              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n ,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n_,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> N ,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K ,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K_,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> S
){

    __shared__ double _H1[size_x][3][3];
    __shared__ double _H2[size_x][3][3];

    __shared__ double _H1inv[size_x][3][3];
    __shared__ double _H2inv[size_x][3][3]; 

    __shared__ double _H1invT[size_x][3][3]; 
    __shared__ double _H2invT[size_x][3][3]; 

    __shared__ double _N1[size_x][3][3]; 
    __shared__ double _N2[size_x][3][3]; 

    __shared__ double _S_[size_x][3][3];
    __shared__ double _ST[size_x][3][3];
    __shared__ double _T0[size_x][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idx = threadIdx.x; 
    const unsigned int idy = threadIdx.y; 
    const unsigned int idz = threadIdx.z; 
    if (_idx >= S.size({0})){return;}
    double h1_i  = H1_inv[_idx][idy][idz]; 
    double h2_i  = H2_inv[_idx][idy][idz]; 

    double circ = (idy == idz)*(2*(idy < 2) - 1); 
    double dmet = met_xy[_idx][idy] * (idz == 2 && idy < 2) - circ; 

    S[_idx][idy][idz]  = dmet; 

    _T0[idx][idy][idz] = circ;
    _H1inv[idx][idy][idz]  = h1_i;
    _H2inv[idx][idy][idz]  = h2_i;

    _H1invT[idx][idz][idy] = h1_i;
    _H2invT[idx][idz][idy] = h2_i;

    __syncthreads();
    h1_i = H1[_idx][idy][idz]; 
    h2_i = H2[_idx][idy][idz]; 

    _H1[idx][idy][idz] = _dot(_H1invT[idx], _T0[idx], idy, idz, 3); 
    _H2[idx][idy][idz] = _dot(_H2invT[idx], _T0[idx], idy, idz, 3); 
    __syncthreads(); 

    _N1[idx][idy][idz] = _dot(_H1[idx], _H1inv[idx], idy, idz, 3); 
    _N2[idx][idy][idz] = _dot(_H2[idx], _H2inv[idx], idy, idz, 3); 
    N[_idx][idy][idz]  = _N1[idx][idy][idz]; 

    _S_[idx][idy][idz] = dmet; 
    _ST[idx][idz][idy] = dmet; 
    __syncthreads(); 

    _H1[idx][idy][idz] = h1_i; 
    _H2[idx][idy][idz] = h2_i; 

    _H1invT[idx][idy][idz] = _dot(_ST[idx], _N1[idx], idy, idz, 3); 
    _H2invT[idx][idy][idz] = _dot(_ST[idx], _N2[idx], idy, idz, 3); 
    __syncthreads(); 

    n [_idx][idy][idz] = _dot(_H1invT[idx], _S_[idx], idy, idz, 3);
    n_[_idx][idy][idz] = _dot(_H2invT[idx], _S_[idx], idy, idz, 3); 

    K [_idx][idy][idz] = _dot(_H1[idx], _H1inv[idx], idy, idz, 3); 
    K_[_idx][idy][idz] = _dot(_H2[idx], _H2inv[idx], idy, idz, 3); 
}

template <typename scalar_t, size_t size_x>
__global__ void _nunu_vp_(
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> S, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> K_,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> n_,

              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> v_,
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> ds,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu0,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu1
){
    __shared__ double _S_[size_x][12][3]; 

    __shared__ double _K0[size_x][6][3]; 
    __shared__ double _K1[size_x][6][3]; 

    __shared__ double _v0[size_x][6][3]; 
    __shared__ double _v1[size_x][6][3]; 
    
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idx = threadIdx.x; 
    const unsigned int idy = threadIdx.y; 
    const unsigned int idz = threadIdx.z; 
    const unsigned int jdz = threadIdx.y%3; 
    if (_idx >= S.size({0})){return;}

    _S_[idx][jdz][idz] = S[_idx][jdz][idz];  
    _v0[idx][idy][idz] = v[_idx][idy][idz];

    _K0[idx][jdz  ][idz] = K[_idx][jdz][idz]; 
    _K0[idx][jdz+3][idz] = n[_idx][jdz][idz]; 

    _K1[idx][jdz  ][idz] = K_[_idx][jdz][idz]; 
    _K1[idx][jdz+3][idz] = n_[_idx][jdz][idz]; 
    __syncthreads(); 

    ds[_idx][idy] = 0; 
    v[_idx][idy][idz] = 0; 

    _v1[idx][idy][idz] = _dot(_S_[idx][idz], _v0[idx][idy], 3); 
    __syncthreads();
    double nu0_ = _dot(_K0[idx][idz], _v0[idx][idy], 3); 
    double nu1_ = _dot(_K1[idx][idz], _v1[idx][idy], 3); 

    _S_[idx][idy  ][idz] = _dot(_K1[idx][idz+3], _v0[idx][idy], 3); 
    _S_[idx][idy+6][idz] = _dot(_K0[idx][idz+3], _v1[idx][idy], 3); 
    __syncthreads(); 

    double dq = _dot(_S_[idx][idy], _v0[idx][idy], 3) - _dot(_S_[idx][idy+6], _v1[idx][idy], 3);  
    dq = log10(dq*dq + (dq == 0)); 
    _K0[idx][idy][idz] = dq;  
    __syncthreads(); 

    int pos = 0; 
    for (size_t y(0); y < 6; ++y){
        if (!_K0[idx][y][idz]){continue;}
        bool kl = (dq > _K0[idx][y][idz]); 
        pos += kl;
    }
    if (!dq){return;}
    v [_idx][pos][idz] = _v0[idx][idy][idz]; 
    v_[_idx][pos][idz] = _v1[idx][idy][idz]; 

    nu0[_idx][pos][idz] = nu0_;
    nu1[_idx][pos][idz] = nu1_; 
    ds[_idx][pos] = dq; 
}

template <typename scalar_t, size_t size_x, size_t size_y>
__global__ void _residual_(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> metxy, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp_,
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu1, 
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> nu2, 
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> dst, 
              const double tol, const double step, const unsigned int timeout
){

    __shared__ double _metxy_[size_x][3]; 
    __shared__ double _H_perp[size_x][6][3]; 

    __shared__ double _res_[size_x][8][8][3]; 
    __shared__ double _Jxb_[size_x][size_y][3];
    __shared__ double _params[size_x][size_y][3]; 
    __shared__ double _buffer[size_x][size_y][2][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idx = threadIdx.x; 
    const unsigned int idy = threadIdx.y;
    const unsigned int idz = threadIdx.z;
    if (_idx >= metxy.size({0})){return;}

    if (idy < 6){_H_perp[idx][idy][idz] = (idy <= 2)*H_perp [_idx][idy%3][idz] + (idy <= 5 && idy >= 3)*H_perp_[_idx][(idy-3)%3][idz];}
    _metxy_[idx][idz] = metxy[_idx][idz*(idz < 2)]*(idz < 2) + 1*(idz == 2); 
    
    int ix = idy % 8; 
    int iy = idy / 8;
    double dt0 = double(4 - ix)*step; 
    double dt1 = double(4 - iy)*step; 
    double v1(0), v2(0), r2c(0); 

    _params[idx][idy][idz] = 0;
    _res_[idx][ix][iy][idz] = 0; 
    _buffer[idx][idy][0][idz] = 0; 
    _buffer[idx][idy][1][idz] = 0; 
    __syncthreads(); 

    for (unsigned int t(0); t < timeout; ++t){
        _buffer[idx][idy][0][idz] = foptim(_params[idx][idy][0] + dt0, idz); 
        _buffer[idx][idy][1][idz] = foptim(_params[idx][idy][1] + dt1, idz); 
        __syncthreads(); 

        v1 = _dot(_H_perp[idx][idz  ], _buffer[idx][idy][0], 3); 
        v2 = _dot(_H_perp[idx][idz+3], _buffer[idx][idy][1], 3);
        r2c = _params[idx][idy][2]; 
        if ( (r2c < tol || t >= timeout - 1) && t ){break;}
        _res_[idx][iy][ix][idz] = pow(v1 + v2 - _metxy_[idx][idz], 2); 

        __syncthreads(); 
        double r2_t  = _sum(_res_[idx][4 ][4 ], 2); 
        double r2_dx = _sum(_res_[idx][iy][ix], 2);
        double r2_dy = _sum(_res_[idx][iy][ix], 2); 

        double gr_t0 = (r2_dx - r2_t)*_div(dt0);
        double gr_t1 = (r2_dy - r2_t)*_div(dt1); 
        double dotx  = gr_t0*gr_t0 + gr_t1*gr_t1; 
        dotx = dotx - _cmp(abs(dt0), abs(dt1), (r2_dx - r2_t)*(r2_dy - r2_t));

        _buffer[idx][idy][0][idz] = trigger(idz < 2, _params[idx][idy][idz], r2_t); 
        _buffer[idx][idy][1][idz] = trigger(idz, r2_dx, r2_dy, r2_t); 
        _Jxb_[idx][idy][idz] = trigger(idz, gr_t0, gr_t1, dotx);

        __syncthreads(); 
        unsigned int py(0), pz(0); 
        for (unsigned int y(0); y < size_y; ++y){
            for (unsigned int z(0); z < 3; ++z){pz = trigger(_buffer[idx][py][1][pz] < _buffer[idx][y][1][z], pz, z);}
            py = trigger(_buffer[idx][py][1][pz] < _buffer[idx][y][1][pz], py, y);
        }
        _params[idx][idy][idz] = _buffer[idx][idy][0][idz] - _div(_Jxb_[idx][py][2]) * _Jxb_[idx][py][idz] * r2_t * (idz < 2);
        __syncthreads(); 
    }

    if (idy == 36 && r2c < tol){
        nu1[_idx][5][idz] = v1; 
        nu2[_idx][5][idz] = v2;
        dst[_idx][5] = log10(r2c);
    }
}

std::map<std::string, torch::Tensor> nusol_::NuNu(
        torch::Tensor* H1_, torch::Tensor* H1_perp, torch::Tensor* H2_, torch::Tensor* H2_perp, torch::Tensor* met_xy,
        double null, const double step, const double tolerance, const unsigned int timeout
){
    const unsigned int dx = H1_ -> size({0}); 
    const unsigned int thx = (dx >= 48) ? 48 : dx; 

    const dim3 thr = dim3(4, 64, 3);
    const dim3 thd = dim3(thx, 3, 3);
    const dim3 thN = dim3(thx, 6, 3);

    const dim3 blr = blk_(dx, 4, 64, 64, 3, 3); 
    const dim3 blk = blk_(dx, thx, 3, 3, 3, 3); 
    const dim3 blN = blk_(dx, thx, 6, 6, 3, 3); 

    torch::Tensor H1_inv = std::get<0>(operators_::Inverse(H1_perp)); 
    torch::Tensor H2_inv = std::get<0>(operators_::Inverse(H2_perp)); 

    torch::Tensor S  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor N  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor n  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor n_ = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor K  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor K_ = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 

    AT_DISPATCH_ALL_TYPES(H1_ -> scalar_type(), "NuNu", [&]{
        _nunu_init_<scalar_t, 48><<<blk, thd>>>(
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                   H1_inv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H2_inv.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H1_ -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H2_ -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),

                        n.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       n_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        N.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        K.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       K_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        S.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    }); 

    std::map<std::string, torch::Tensor> out = nusol_::Intersection(&N, &n_, null); 
    torch::Tensor nu1 = torch::zeros_like(out["solutions"]); 
    torch::Tensor nu2 = torch::zeros_like(out["solutions"]); 
    torch::Tensor v_  = torch::zeros_like(out["solutions"]); 
    torch::Tensor v   = out["solutions"]; 
    torch::Tensor ds  = out["distances"]; 

    AT_DISPATCH_ALL_TYPES(H1_ -> scalar_type(), "NuNu", [&]{
        if (timeout){
            _residual_<scalar_t, 4, 64><<<blr, thr>>>(
                    met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                   H1_perp -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H2_perp -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                            v.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                           v_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                           ds.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                            tolerance, step, timeout
            ); 
        }

        _nunu_vp_<scalar_t, 48><<<blN, thN>>>(
                        S.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        K.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       K_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        n.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       n_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),

                        v.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                       v_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                       ds.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                      nu1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                      nu2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()
        ); 
    }); 

    out["nu1"] = nu1; 
    out["K"] = K; 
    out["n"] = n; 

    out["nu2"] = nu2; 
    out["K_"] = K_; 
    out["n_"] = n_; 
    return out;  
}

std::map<std::string, torch::Tensor> nusol_::NuNu(
            torch::Tensor* pmc_b1,  torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy,  double null, torch::Tensor* m1, torch::Tensor* m2, 
            const double step, const double tolerance, unsigned int timeout
){
    if (!m2){m2 = m1;}
    unsigned int dx = met_xy -> size(0); 
    std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(pmc_b1, pmc_mu1, m1);
    std::map<std::string, torch::Tensor> H2_m = nusol_::BaseMatrix(pmc_b2, pmc_mu2, m2);
    torch::Tensor passed = H1_m["passed"] * H2_m["passed"]; 

    torch::Tensor H1_ = H1_m["H"]; 
    torch::Tensor H1p = H1_m["H_perp"]; 

    torch::Tensor H2_ = H2_m["H"]; 
    torch::Tensor H2p = H2_m["H_perp"]; 

    std::map<std::string, torch::Tensor> out; 
    out = nusol_::NuNu(&H1_, &H1p, &H2_, &H2p, met_xy, null, step, tolerance, timeout); 
    torch::Tensor nu1 = out["nu1"].view({dx, -1, 3});  
    torch::Tensor nu2 = out["nu2"].view({dx, -1, 3}); 
    torch::Tensor dst = out["distances"].view({dx, -1}); 
    out["passed"] = passed; 
    return out; 
}

std::map<std::string, torch::Tensor> nusol_::NuNu(
            torch::Tensor* pmc_b1,  torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy, double null, double massT1, double massW1, double massT2, double massW2
){
    std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(pmc_b1, pmc_mu1, massT1, massW1, 0);
    std::map<std::string, torch::Tensor> H2_m = nusol_::BaseMatrix(pmc_b2, pmc_mu2, massT2, massW2, 0);

    torch::Tensor H1_ = H1_m["H"]; 
    torch::Tensor H1p = H1_m["H_perp"]; 

    torch::Tensor H2_ = H2_m["H"]; 
    torch::Tensor H2p = H2_m["H_perp"]; 

    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&H1_, &H1p, &H2_, &H2p, met_xy, null); 
    out["passed"] = H1_m["passed"] * H2_m["passed"]; 
    return out; 
}




