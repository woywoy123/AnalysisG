#ifndef CU_NUSOL_UTILS_H
#define CU_NUSOL_UTILS_H
#include <utils/atomic.cuh>

template <typename scalar_t>
__global__ void _count(
        const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> batch, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pid, 
              torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> num_events,
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> num_pid,
        const unsigned int dx
){
    extern __shared__ long _batch_idx[]; 
    _batch_idx[threadIdx.x] = batch[threadIdx.x]; 
    _batch_idx[threadIdx.x +   dx] = pid[threadIdx.x][0];
    _batch_idx[threadIdx.x + 2*dx] = pid[threadIdx.x][1];
    __syncthreads(); 
    long id = _batch_idx[threadIdx.x]; 

    long ev = 0; 
    long sx = -1; 
    long num_lep = 0;
    long num_bqk = 0; 
    for (size_t x(0); x < dx; ++x){
        if (!(_batch_idx[x] == id)){continue;} 
        num_lep += (_batch_idx[x + dx  ] == 1); 
        num_bqk += (_batch_idx[x + dx*2] == 1); 
        sx = (sx < 0) ? x : sx; 
        ++ev;
    }
    if (threadIdx.x != sx){return;}
    num_events[id] = ev; 
    num_pid[id][0] = num_lep;
    num_pid[id][1] = num_bqk; 
}

template <typename scalar_t>
__global__ void _combination(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pid,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> batch,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> num_edges,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> edge_idx,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> num_pid,
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> llbb,
          torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> msk,
    const unsigned int n_edges, const unsigned mx
){
    const unsigned int __idx = blockIdx.x * blockDim.x + threadIdx.x; 

    const unsigned int _idx = __idx / mx; 
    long src_i = edge_idx[0][_idx]; 
    long dst_i = edge_idx[1][_idx]; 
    long ev_dx = batch[src_i]; 

    const unsigned int _idy = num_edges[ev_dx] + __idx % mx; 
    if (n_edges <= _idy){return;}

    long src_j = edge_idx[0][_idy]; 
    long dst_j = edge_idx[1][_idy]; 
    long ev_dy = batch[src_j]; 
    if (ev_dx != ev_dy){return;}

    long num_l = num_pid[ev_dx][0]; 
    long num_b = num_pid[ev_dx][1]; 
    if (num_l > 2 || num_b < 1 || num_l == 0){return;}

    bool l_src_i = pid[src_i][0] > 0; 
    bool l_dst_i = pid[dst_i][0] > 0; 
    bool l_src_j = pid[src_j][0] > 0; 
    bool l_dst_j = pid[dst_j][0] > 0; 

    bool b_src_i = pid[src_i][1] > 0;
    bool b_dst_i = pid[dst_i][1] > 0;
    bool b_src_j = pid[src_j][1] > 0; 
    bool b_dst_j = pid[dst_j][1] > 0; 


    // both source and dest are leptons 
    int ll  = (src_i != src_j)*(l_src_i * l_src_j); 

    // both source and dest are b-quarks
    int bb  = (dst_i != dst_j)*(b_dst_i * b_dst_j); 

    // source is lep and dest is b-quark
    int lb  = (src_i != dst_i)*(l_src_i * b_dst_i)*(dst_i != dst_j); 

    // source is lep and dest is b-quark
    int lb_ = (src_j != dst_j)*(l_src_j * b_dst_j)*(src_i != src_j); 

    bool lx = ((ll + bb) == 2) + ((lb + lb_) == num_l);
    if (!lx){return;}
    if (((dst_i != dst_j) + (l_src_i) + (src_i != dst_i)) * (num_l == 1)){return;}

    llbb[__idx][0] = (l_src_i)*src_i - 1 * (!l_src_i); // l1
    llbb[__idx][1] = (l_src_j)*src_j - 1 * (!l_src_j); // l2
    llbb[__idx][2] = (b_dst_i)*dst_i - 1 * (!b_dst_i); // b1
    llbb[__idx][3] = (b_dst_j)*dst_j - 1 * (!b_dst_j); // b2
    llbb[__idx][4] = ev_dx; 
    msk[__idx] = lx*num_l; 
}

__global__ void _mass_matrix(
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> mass_,
        double mTl, double mTs, double mWl, double mWs, unsigned int steps
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (steps <= _idx){return;}
    double lw = mWl*threadIdx.y + mTl*(1 - threadIdx.y); 
    double dx = mWs*threadIdx.y + mTs*(1 - threadIdx.y);
    mass_[_idx][threadIdx.y] = lw + mass_[_idx][threadIdx.y]*dx*_idx; 
}


template <size_t size_x>
__global__ void _perturbation(
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_tw1,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_tw2,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_met,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_res,
        const double perturb, const double top_mass, 
        const double w_mass, const bool start
){
    __shared__ double _dnu_res[size_x][36][6]; 
    __shared__ double _dnu_par[size_x][36][6]; 

    const unsigned int idx  = threadIdx.x;
    const unsigned int idy  = threadIdx.y; 
    const unsigned int idz  = threadIdx.z;  
    const unsigned int _idx = (blockIdx.x * blockDim.x + threadIdx.x)*36 + idy; 
    if (_idx >= dnu_res.size({0})){return;}

    if (idz < 2){     _dnu_par[idx][idy][idz] = dnu_tw1[_idx][idz] * top_mass;}
    else if (idz < 4){_dnu_par[idx][idy][idz] = dnu_tw2[_idx][idz] * w_mass;}
    else if (idz < 6){_dnu_par[idx][idy][idz] = dnu_met[_idx][idz];}
    _dnu_res[idx][idy][idz] = dnu_res[_idx][idz]; 
    __syncthreads();

}

__global__ void _assign_mass(
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> mass_tw,
        const double mass_t, const double mass_w, const long lenx
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const bool _idy = threadIdx.y > 0; 
    if (_idx >= lenx){return;}
    mass_tw[_idx][threadIdx.y] = mass_t * (1 - _idy) + mass_w*_idy; 
}

template <size_t size_x>
__global__ void _perturb(
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> nu_params,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_met,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_tw1,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_tw2,
        const unsigned long lnx, const double dt, const unsigned int ofs
){
    __shared__ double _params_[size_x][6]; 

    const unsigned int _idx = threadIdx.x; 
    const unsigned int _idy = threadIdx.y; 
    const unsigned int _idz = threadIdx.z;  
    const unsigned int idx  = blockIdx.x * blockDim.x + _idx; 
    if (idx >= lnx){return;}
    if (!_idy){_params_[_idx][_idz] = nu_params[idx][_idz];}
    const unsigned int _idt = idx * ofs + _idy; 
    __syncthreads(); 

    double dx_ = _params_[_idx][_idz] + (_idy == _idz) * dt; 
    if (_idz < 2){dnu_met[_idt][_idz  ] = dx_; return;}
    if (_idz < 4){dnu_tw1[_idt][_idz-2] = dx_; return;}
    if (_idz < 6){dnu_tw2[_idt][_idz-4] = dx_; return;}
}

template <size_t size_x>
__global__ void _jacobi(
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> nu_params,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> dnu_res,
        const unsigned long lnx, const double dt, unsigned int ofs
){
    __shared__ double _jacobi_ [size_x][6]; 
    __shared__ double _Jxt_    [size_x][6]; 
    __shared__ double _param_  [size_x][7]; 

    const unsigned int _idx = threadIdx.x; 
    const unsigned int _idy = threadIdx.y; 
    const unsigned int _idp = blockIdx.x * blockDim.x + _idx; 
    const unsigned int idx  = (blockIdx.x * blockDim.x + _idx  )*ofs + _idy; 
    if (idx >= lnx*ofs){return;}

    if (_idy == 1){_param_[_idx][6] = dnu_res[(blockIdx.x * blockDim.x + _idx+1)*ofs-1][0];}
    _param_[_idx][_idy] = nu_params[_idp][_idy];
    __syncthreads(); 

    double nu_x0 = _param_[_idx][6]; 
    _jacobi_[_idx][_idy] = (dnu_res[idx][0] - nu_x0)/dt;
    _Jxt_   [_idx][_idy] = _jacobi_[_idx][_idy] * _jacobi_[_idx][_idy]; 
    __syncthreads();

    double dv = _div(_sum(_Jxt_[_idx], 6)) * _jacobi_[_idx][_idy] * nu_x0; 
    nu_params[_idp][_idy] = _param_[_idx][_idy] - dv;  
}

template <size_t size_x>
__global__ void _compare_solx(
        torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> cmx_dx,
        torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> evnt_dx,

        torch::PackedTensorAccessor64<double, 1, torch::RestrictPtrTraits> o_sol,
        torch::PackedTensorAccessor64<long  , 2, torch::RestrictPtrTraits> o_cmb,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> o_nu1,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> o_nu2,

        torch::PackedTensorAccessor64<long  , 2, torch::RestrictPtrTraits> i_cmb,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> i_sol,
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> i_nu1,
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> i_nu2, 
        const unsigned int lx, const unsigned int evnts
){

    __shared__ double _nu1[size_x][4]; 
    __shared__ double _nu2[size_x][4]; 
    __shared__ double _cmx[size_x][4]; 
    __shared__ double _score[size_x][4]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (_idx >= evnts){return;}

    for (size_t x(0); x < lx; ++x){
        long cmx_id = cmx_dx[x]; 
        long evn_id = evnt_dx[cmx_id]; 
        if (evn_id != _idx){continue;}
        for (size_t y(0); y < 6; ++y){
            double sol = i_sol[x][y]; 
            if (_score[threadIdx.x][threadIdx.y] < sol){continue;}
            _score[threadIdx.x][threadIdx.y] = sol; 
            _cmx[threadIdx.x][threadIdx.y] = i_cmb[cmx_id][threadIdx.y]; 
            _nu1[threadIdx.x][threadIdx.y] = (threadIdx.y < 3) ? i_nu1[x][y][threadIdx.y] : 0; 
            _nu2[threadIdx.x][threadIdx.y] = (threadIdx.y < 3) ? i_nu2[x][y][threadIdx.y] : 0; 
        }
    }

    __syncthreads(); 
    o_sol[_idx] = _score[threadIdx.x][threadIdx.y]; 
    o_cmb[_idx][threadIdx.y] = _cmx[threadIdx.x][threadIdx.y]; 
    o_nu1[_idx][threadIdx.y] = (threadIdx.y < 3) ? _nu1[threadIdx.x][threadIdx.y] : _sqrt(_dot(_nu1[threadIdx.x], _nu1[threadIdx.x], 3)); 
    o_nu2[_idx][threadIdx.y] = (threadIdx.y < 3) ? _nu2[threadIdx.x][threadIdx.y] : _sqrt(_dot(_nu2[threadIdx.x], _nu2[threadIdx.x], 3)); 
}

#endif
