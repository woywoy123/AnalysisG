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
    torch::Tensor n  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor n_ = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor K  = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 
    torch::Tensor K_ = torch::zeros({dx, 3, 3}, MakeOp(H1_)); 

    AT_DISPATCH_ALL_TYPES(H1_ -> scalar_type(), "NuNu", [&]{
        _nunu_init_<scalar_t, 48><<<blk, thd>>>(
                met_xy -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                H1_inv -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                H2_inv -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H1_ -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                   H2_ -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),

                        n.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
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

    const dim3 thN = dim3(thx, 6, 3);
    const dim3 blN = blk_(dx, thx, 6, 6, 3, 3); 
    AT_DISPATCH_ALL_TYPES(H1_ -> scalar_type(), "NuNu", [&]{
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
                      nu2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>()); 
    }); 

    out["nu1"] = nu1; 
    out["K"] = K; 
    out["n"] = n; 

    out["nu2"] = nu2; 
    out["K_"] = K_; 
    out["n_"] = n_; 
    return out;  
}


std::vector<torch::Tensor> residuals(
        torch::Tensor* H_perp, torch::Tensor* H_perp_, torch::Tensor* met_xy, 
        torch::Tensor resid, double limit = 1e-5, int max_iter = 10000
){
    //pybind11::gil_scoped_release no_gil; 
    torch::Tensor _H_perp  = H_perp  -> index({resid}); 
    torch::Tensor _H_perp_ = H_perp_ -> index({resid}); 
    torch::Tensor _met     = met_xy -> index({resid}); 
    unsigned int dx = _met.size(0); 
    if (dx == met_xy -> size(0)){ return {}; }
	
    torch::Tensor t1 = torch::ones( {dx, 1}, MakeOp(H_perp));
    torch::Tensor t0 = torch::zeros({dx, 1}, MakeOp(H_perp)); 
    torch::Tensor pi = torch::cos(t1)*2; 
    _met = torch::cat({_met, t1}, -1);
    t0 = torch::cat({t0, t0}, -1); 
    
    torch::Tensor t  = torch::zeros({dx, 1}, MakeOp(H_perp).requires_grad(true)); 
    torch::Tensor t_ = torch::zeros({dx, 1}, MakeOp(H_perp).requires_grad(true));
    torch::nn::functional::MSELossFuncOptions fx(torch::kNone); 
    
    torch::optim::AdamOptions set(0.001); 
    torch::optim::Adam opti({t, t_}, set); 

    torch::Tensor nu, nu_, loss, l1; 
    for (int i(0); i < max_iter; ++i){
        torch::Tensor px_ = pi*t; 
        torch::Tensor py_ = pi*t_; 

        nu  = ( _H_perp * (torch::cat({torch::cos(px_), torch::sin(px_), t1}, -1).view({-1, 1, 3}))).sum(-1); 
	nu_ = (_H_perp_ * (torch::cat({torch::cos(py_), torch::sin(py_), t1}, -1).view({-1, 1, 3}))).sum(-1);
        torch::Tensor nus = (nu_ + nu - _met).index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 2)});
	opti.zero_grad(); 

	loss = torch::nn::functional::mse_loss(nus, t0, fx); 
	loss.sum().backward(); 
	opti.step();
        if (!i){l1 = loss.detach(); continue;}
        if ((torch::abs(l1 - loss).sum(-1) < 1e-12).sum(-1).item<bool>()){break;}
        l1 = loss.detach(); 
        if (!t0.index({loss < limit}).size({0})){continue;}
    }
    return {nu.detach().view({-1, 1, 1, 3}), nu_.detach().view({-1, 1, 1, 3}), torch::log10(loss.detach().sum(-1))};  
}


std::map<std::string, torch::Tensor> nusol_::NuNu(
            torch::Tensor* pmc_b1,  torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy,  double null, torch::Tensor* m1, torch::Tensor* m2
){
    if (!m2){m2 = m1;}
    std::map<std::string, torch::Tensor> H1_m = nusol_::BaseMatrix(pmc_b1, pmc_mu1, m1);
    std::map<std::string, torch::Tensor> H2_m = nusol_::BaseMatrix(pmc_b2, pmc_mu2, m2);
    torch::Tensor passed = H1_m["passed"] * H2_m["passed"]; 

    torch::Tensor H1_inv = std::get<0>(operators_::Inverse(&H1_m["H_perp"])); 
    torch::Tensor H2_inv = std::get<0>(operators_::Inverse(&H2_m["H_perp"])); 

    torch::Tensor H1_    = H1_m["H"]; 
    torch::Tensor H2_    = H2_m["H"]; 

    std::map<std::string, torch::Tensor> out = nusol_::NuNu(&H1_, &H1_inv, &H2_, &H2_inv, met_xy, null); 

    //unsigned int dx = met_xy -> size(0); 
    //torch::Tensor nu1 = out["nu1"].view({dx, -1, 3});  
    //torch::Tensor nu2 = out["nu2"].view({dx, -1, 3}); 
    //torch::Tensor dst = out["distances"].view({dx, -1}); 

    //torch::Tensor msk = ((dst != 0).sum(-1) != 0)*(passed == 1); 
    //std::vector<torch::Tensor> mx = residuals(&H1_m["H_perp"], &H2_m["H_perp"], met_xy, msk); 

    //if (mx.size()){
    //    unsigned int lx = msk.index({msk}).size(0);
    //    torch::Tensor nu1_ = ( out["K"].index({msk}).view({lx, 1, 3, 3}) * mx[0]).sum(-1).view({-1, 1, 3}); 
    //    torch::Tensor nu2_ = (out["K_"].index({msk}).view({lx, 1, 3, 3}) * mx[1]).sum(-1).view({-1, 1, 3});
    //    torch::Tensor nullx = torch::zeros_like(nu1_); 
    //    torch::Tensor nulld = torch::zeros({lx, 5}, MakeOp(pmc_mu1)); 

    //    std::vector<torch::Tensor> vx = {}; 
    //    for (size_t x(0); x < 5; ++x){vx.push_back(nullx);}
    //    nullx = torch::cat(vx, -1); 
    //    
    //    nu1.index_put_({msk}, torch::cat({nu1_, nullx}, -1).view({-1, 6, 3}));  
    //    nu2.index_put_({msk}, torch::cat({nu2_, nullx}, -1).view({-1, 6, 3}));  
    //    dst.index_put_({msk}, torch::cat({mx[2].view({-1, 1}), nulld}, -1).view({lx, -1})); 
    //}
    //out["nu1"] = nu1; 
    //out["nu2"] = nu2; 
    out["passed"] = passed; 
    //out["distances"] = dst; 
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




