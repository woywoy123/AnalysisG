#include <nusol/base.cuh>
#include <nusol/device.cuh>
#include <utils/utils.cuh>
#include <physics/physics.cuh>
#include <operators/operators.cuh>
#include <transform/transform.cuh>

template <typename scalar_t>
__global__ void _hmatrix_debug(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> masses, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> cosine, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> rt,

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_l, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2l, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2l, 

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2b, 

        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Hmatrix,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp ,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> passed,

        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x0p,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>  x0,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>  Sx,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>  Sy,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>   w,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> om2,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> eps2,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>  x1,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>  y1,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits>  z
){
    __shared__ double rotT[3][3]; 
    __shared__ double Htil[4][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    const unsigned int _iky = (_idy * 4 + _idz)/4; 
    const unsigned int _ikz = (_idy * 4 + _idz)%4; 

    nusol sl = nusol(); 
    sl.cos = cosine[_idx][0]; 
    sl.betas[0] = b2l[_idx][0]; 
    sl.pmass[0] = m2l[_idx][0]; 

    sl.betas[1] = b2b[_idx][0]; 
    sl.pmass[1] = m2b[_idx][0]; 
    for (size_t x(0); x < 4; ++x){
        sl.pmu_b[x] = pmc_b[_idx][x];
        sl.pmu_l[x] = pmc_l[_idx][x];
        if (x > 2){continue;}
        sl.masses[x] = masses[_idx][x];
    } 
    _makeNuSol(&sl); 
    if (_iky < 3 && _ikz < 3){rotT[_iky][_ikz] = rt[_idx][_iky][_ikz];}
    Htil[_iky][_ikz] = _htilde(&sl, _iky, _ikz);  
    __syncthreads(); 
    if (_iky < 3 && _ikz < 3){
        double hx = _dot(rotT, Htil, _iky, _ikz, 3); 
        Hmatrix[_idx][_iky][_ikz] = hx;
        H_perp[_idx][_iky][_ikz] = (_iky < 2) ? hx : _ikz == 2; 
    }
    if (threadIdx.y || threadIdx.z){return;}
    passed[_idx] = sl.passed; 
    x0p[_idx]    = sl.x0p;
    x0[_idx]     = sl.x0;
    Sx[_idx]     = sl.sx;
    Sy[_idx]     = sl.sy;
    w[_idx]      = sl.w;
    om2[_idx]    = sl.o2;
    eps2[_idx]   = sl.eps2;
    x1[_idx]     = sl.x1;
    y1[_idx]     = sl.y1;
    z[_idx]      = sl.z;
}


std::map<std::string, torch::Tensor> nusol_::BaseDebug(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses){
    torch::Tensor b2_b  = physics_::Beta2(pmc_b);
    torch::Tensor b2_l  = physics_::Beta2(pmc_mu); 
    torch::Tensor m2_b  = physics_::M2(pmc_b);
    torch::Tensor m2_l  = physics_::M2(pmc_mu); 
    torch::Tensor theta = physics_::Theta(pmc_mu); 
    torch::Tensor cthe  = operators_::CosTheta(pmc_b, pmc_mu, 3); 

    torch::Tensor px    = pmc_mu -> index({torch::indexing::Slice(), 0});
    torch::Tensor py    = pmc_mu -> index({torch::indexing::Slice(), 1}); 
    torch::Tensor phi   = transform_::Phi(&px, &py);
    torch::Tensor rt    = operators_::RT(pmc_b, &phi, &theta); 

    const unsigned int dx = pmc_b -> size({0}); 
    const dim3 thd = dim3(1, 4, 4);
    const dim3 blk = blk_(dx, 1, 4, 4, 4, 4); 

    torch::Tensor HMatrix = torch::zeros({dx, 3, 3}, MakeOp(pmc_mu)); 
    torch::Tensor H_perp  = torch::zeros({dx, 3, 3}, MakeOp(pmc_mu)); 
    torch::Tensor passed  = torch::zeros({dx}, MakeOp(pmc_mu)); 

    torch::Tensor x0p     = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor x0      = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor Sx      = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor Sy      = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor w       = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor om2     = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor eps2    = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor x1      = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor y1      = torch::zeros({dx}, MakeOp(pmc_mu)); 
    torch::Tensor z       = torch::zeros({dx}, MakeOp(pmc_mu)); 


    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "BaseMatrix", [&]{
        _hmatrix_debug<scalar_t><<<blk, thd>>>(
                masses -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     cthe.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                       rt.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),

                pmc_mu -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     m2_l.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     b2_l.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),

                pmc_b -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    m2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    b2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),

                // ---------- outputs ---------- //
                HMatrix.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 H_perp.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 passed.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
        
                    x0p.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
                     x0.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     Sx.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     Sy.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                      w.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                    om2.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                   eps2.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     x1.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                     y1.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
                      z.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>()); 
    }); 

    std::map<std::string, torch::Tensor> out; 
    out["H"]      = HMatrix; 
    out["H_perp"] = H_perp; 
    out["passed"] = passed; 
    out["x0p"]  =  x0p;
    out["x0"]   =   x0;
    out["Sx"]   =   Sx;
    out["Sy"]   =   Sy;
    out["w"]    =    w;
    out["om2"]  =  om2;
    out["eps2"] = eps2;
    out["x1"]   =   x1;
    out["y1"]   =   y1;
    out["z"]    =    z;
    out["rt"]   =   rt;
    return out; 
}

template <size_t size_x>
__global__ void _hmatrix(
        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> masses, 
        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> cosine, 
        const torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> rt,

        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> pmc_l, 
        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> m2l, 
        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> b2l, 

        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> pmc_b, 
        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> m2b, 
        const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> b2b, 

        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> Hmatrix,
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> H_perp,
        torch::PackedTensorAccessor64<double, 1, torch::RestrictPtrTraits> passed, 
        const unsigned int dy, const unsigned int dx, const bool sm
){
    __shared__ double rotT[size_x][3][3]; 
    __shared__ double Htil[size_x][3][3];

    const unsigned int idx  = threadIdx.x;
    const unsigned int idy  = threadIdx.y;
    const unsigned int idz  = threadIdx.z; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _lim = (dy*dx)*(!sm) + sm*dx; 
    const unsigned int _ikx = _idx / ((!sm)*dy + sm);
    const unsigned int _ikm = _idx % dy;  
    if (_idx >= _lim){return;}

    nusol sl = nusol(); 
    sl.cos = cosine[_ikx][0]; 
    sl.betas[0] = b2l[_ikx][0]; 
    sl.pmass[0] = m2l[_ikx][0]; 

    sl.betas[1] = b2b[_ikx][0]; 
    sl.pmass[1] = m2b[_ikx][0]; 
    for (size_t x(0); x < 4; ++x){
        sl.pmu_b[x] = pmc_b[_ikx][x];
        sl.pmu_l[x] = pmc_l[_ikx][x];
    } 
    sl.masses[0] = masses[_ikm][0];
    sl.masses[1] = masses[_ikm][1]; 
    sl.masses[2] = 0; 

    _makeNuSol(&sl); 
    rotT[idx][idy][idz] = rt[_ikx][idy][idz];
    Htil[idx][idy][idz] = _htilde(&sl, idy, idz);  
    __syncthreads(); 

    double hx = _dot(rotT[idx], Htil[idx], idy, idz, 3); 
    Hmatrix[_idx][idy][idz] = sl.passed*hx;
    H_perp[_idx][idy][idz] = sl.passed*((idy < 2) ? hx : idz == 2); 
    if (idy || idz){return;}
    passed[_idx] = sl.passed; 
}

std::map<std::string, torch::Tensor> nusol_::BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses){
    torch::Tensor b2_b  = physics_::Beta2(pmc_b);
    torch::Tensor b2_l  = physics_::Beta2(pmc_mu); 
    torch::Tensor m2_b  = physics_::M2(pmc_b);
    torch::Tensor m2_l  = physics_::M2(pmc_mu); 
    torch::Tensor theta = physics_::Theta(pmc_mu); 
    torch::Tensor cthe  = operators_::CosTheta(pmc_b, pmc_mu, 3); 

    torch::Tensor px    = pmc_mu -> index({torch::indexing::Slice(), 0});
    torch::Tensor py    = pmc_mu -> index({torch::indexing::Slice(), 1}); 
    torch::Tensor phi   = transform_::Phi(&px, &py);
    torch::Tensor rt    = operators_::RT(pmc_b, &phi, &theta); 

    const unsigned int dy = masses -> size({0}); 
    const unsigned int dx = pmc_b -> size({0}); 
    const unsigned int px_ = (dy == dx) ? dx : dx * dy; 
    const dim3 thd = dim3(64, 3, 3);
    const dim3 blk = blk_(px_, 64, 3, 3, 3, 3); 
    torch::Tensor HMatrix = torch::zeros({px_, 3, 3}, MakeOp(pmc_mu)); 
    torch::Tensor H_perp  = torch::zeros({px_, 3, 3}, MakeOp(pmc_mu)); 
    torch::Tensor passed  = torch::zeros({px_}, MakeOp(pmc_mu)); 

    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "BaseMatrix", [&]{
        _hmatrix<64><<<blk, thd>>>(
                masses -> packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                     cthe.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                       rt.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),

                pmc_mu -> packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                     m2_l.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                     b2_l.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),

                pmc_b -> packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    m2_b.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                    b2_b.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),

                // ---------- outputs ---------- //
                 HMatrix.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                  H_perp.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                  passed.packed_accessor64<double, 1, torch::RestrictPtrTraits>(), 
                  dy, dx, dy == dx);
    }); 

    std::map<std::string, torch::Tensor> out; 
    out["H"]      = HMatrix; 
    out["H_perp"] = H_perp; 
    out["passed"] = passed; 
    return out; 
}


template <typename scalar_t>
__global__ void _hmatrix(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> cosine, 
        const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> rt,

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_l, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2l, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2l, 

        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmc_b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> m2b, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> b2b, 

        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> Hmatrix,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> H_perp,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> passed,
        double mass_T, double mass_W, double mass_nu = 0
){
    __shared__ double rotT[3][3]; 
    __shared__ double Htil[4][4];

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    const unsigned int _iky = (_idy * 4 + _idz)/4; 
    const unsigned int _ikz = (_idy * 4 + _idz)%4; 

    nusol sl = nusol(); 
    sl.cos = cosine[_idx][0]; 
    sl.betas[0] = b2l[_idx][0]; 
    sl.pmass[0] = m2l[_idx][0]; 

    sl.betas[1] = b2b[_idx][0]; 
    sl.pmass[1] = m2b[_idx][0]; 
    sl.masses[0] = mass_T; 
    sl.masses[1] = mass_W; 
    sl.masses[2] = mass_nu; 
    for (size_t x(0); x < 4; ++x){
        sl.pmu_b[x] = pmc_b[_idx][x];
        sl.pmu_l[x] = pmc_l[_idx][x];
    } 
    _makeNuSol(&sl); 
    if (_iky < 3 && _ikz < 3){rotT[_iky][_ikz] = rt[_idx][_iky][_ikz];}
    Htil[_iky][_ikz] = _htilde(&sl, _iky, _ikz);  
    __syncthreads(); 
    if (_iky < 3 && _ikz < 3){
        double hx = _dot(rotT, Htil, _iky, _ikz, 3); 
        Hmatrix[_idx][_iky][_ikz] = sl.passed*hx;
        H_perp[_idx][_iky][_ikz] = sl.passed*((_iky < 2) ? hx : _ikz == 2); 
    }
    if (threadIdx.y || threadIdx.z){return;}
    passed[_idx] = sl.passed; 
}

std::map<std::string, torch::Tensor> nusol_::BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, double mT, double mW, double mN){
    torch::Tensor b2_b  = physics_::Beta2(pmc_b);
    torch::Tensor b2_l  = physics_::Beta2(pmc_mu); 
    torch::Tensor m2_b  = physics_::M2(pmc_b);
    torch::Tensor m2_l  = physics_::M2(pmc_mu); 
    torch::Tensor theta = physics_::Theta(pmc_mu); 
    torch::Tensor cthe  = operators_::CosTheta(pmc_b, pmc_mu, 3); 

    torch::Tensor px    = pmc_mu -> index({torch::indexing::Slice(), 0});
    torch::Tensor py    = pmc_mu -> index({torch::indexing::Slice(), 1}); 
    torch::Tensor phi   = transform_::Phi(&px, &py);
    torch::Tensor rt    = operators_::RT(pmc_b, &phi, &theta); 

    const unsigned int dx = pmc_b -> size({0}); 
    const dim3 thd = dim3(1, 4, 4);
    const dim3 blk = blk_(dx, 1, 4, 4, 4, 4); 

    torch::Tensor HMatrix = torch::zeros({dx, 3, 3}, MakeOp(pmc_mu)); 
    torch::Tensor H_perp  = torch::zeros({dx, 3, 3}, MakeOp(pmc_mu)); 
    torch::Tensor KMatrix = torch::zeros({dx, 4, 4}, MakeOp(pmc_mu)); 
    torch::Tensor A_leps  = torch::zeros({dx, 4, 4}, MakeOp(pmc_mu)); 
    torch::Tensor A_bqrk  = torch::zeros({dx, 4, 4}, MakeOp(pmc_mu)); 
    torch::Tensor passed  = torch::zeros({dx}, MakeOp(pmc_mu)); 

    AT_DISPATCH_ALL_TYPES(pmc_b -> scalar_type(), "BaseMatrix", [&]{
        _hmatrix<scalar_t><<<blk, thd>>>(
                cthe.packed_accessor64<scalar_t     , 2, torch::RestrictPtrTraits>(),
                rt.packed_accessor64<scalar_t       , 3, torch::RestrictPtrTraits>(),

                pmc_mu -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                m2_l.packed_accessor64<scalar_t     , 2, torch::RestrictPtrTraits>(),
                b2_l.packed_accessor64<scalar_t     , 2, torch::RestrictPtrTraits>(),

                pmc_b -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                m2_b.packed_accessor64<scalar_t    , 2, torch::RestrictPtrTraits>(),
                b2_b.packed_accessor64<scalar_t    , 2, torch::RestrictPtrTraits>(),

                // ---------- outputs ---------- //
                HMatrix.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                H_perp.packed_accessor64<scalar_t , 3, torch::RestrictPtrTraits>(),
                passed.packed_accessor64<scalar_t , 1, torch::RestrictPtrTraits>(), 
                mT, mW, mN); 
    }); 

    std::map<std::string, torch::Tensor> out; 
    out["H"] = HMatrix; 
    out["H_perp"] = H_perp; 
    out["passed"] = passed; 
    return out; 
}


