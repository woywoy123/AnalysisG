#ifndef CU_NUSOL_UTILS_H
#define CU_NUSOL_UTILS_H
#include <atomic/cuatomic.cuh>

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
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = _idx*steps + _idy; 
    if (steps*steps <= _idz){return;}
    mass_[_idz][0] = mTl + abs(mass_[_idz][0]) * mTs * _idx; 
    mass_[_idz][1] = mWl + abs(mass_[_idz][1]) * mWs * _idy; 
}

__global__ void _combination_matrix(
        torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> indx, 
        const unsigned int ds
){
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    if (_idx >= indx.size({0})){return;}
    long val = 1; 
    for (size_t x(0); x < _idy; ++x){val *= ds;}
    indx[_idx][1 - _idy] = (_idx / val)%ds;  
}


__global__ void _compare_solx(
        torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> evnt_dx,
        torch::PackedTensorAccessor64<double, 1, torch::RestrictPtrTraits> cur_sol,
        torch::PackedTensorAccessor64<long  , 2, torch::RestrictPtrTraits> cur_cmb,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> nu1,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> nu2,
        torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> cmx,

        torch::PackedTensorAccessor64<long  , 2, torch::RestrictPtrTraits> new_cmb,
        torch::PackedTensorAccessor64<double, 1, torch::RestrictPtrTraits> new_sol,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> nu1_,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> nu2_, 
        const unsigned int lx, const unsigned int n_msk, const unsigned int n_evn
){
    extern __shared__ double _inpt[][4];  
    const unsigned int _idx = threadIdx.x; 
    const unsigned int _idy = threadIdx.y; 
    const unsigned int _idz = _idx*n_msk + _idy; 

    _inpt[_idz][0] = 0; 
    _inpt[_idz][1] = -1;
    _inpt[_idz][2] = -1; 
    _inpt[_idz][3] = -1; 
    for (size_t x(0); x < lx; ++x){
        unsigned int edx = evnt_dx[x]; 
        if (cmx[x] != threadIdx.y || edx != threadIdx.x){continue;}
        for (size_t j(x*18); j < 18*(x+1); ++j){
            double xol = new_sol[j]; 
            if (_inpt[_idz][0] < xol){continue;}
            _inpt[_idz][0] = xol; 
            _inpt[_idz][1] = j;
            _inpt[_idz][2] = x; 
            _inpt[_idz][3] = edx; 
        } 
    }
    if (_inpt[_idz][1] == -1){return;}
    const double solx = _inpt[_idz][0]; 
    const unsigned int j_ = _inpt[_idz][1]; 
    const unsigned int x_ = _inpt[_idz][2]; 
    const unsigned int edx = evnt_dx[x_]; 
    for (size_t t(0); t < n_evn*n_msk; ++t){
        if (_inpt[t][3] != edx){continue;}
        if (_inpt[t][0] < solx){return;}
    }

    const unsigned int cx = cmx[x_]; 
    cur_cmb[edx][0] = new_cmb[cx][0]; 
    cur_cmb[edx][1] = new_cmb[cx][1]; 
    cur_cmb[edx][2] = new_cmb[cx][2]; 
    cur_cmb[edx][3] = new_cmb[cx][3]; 

    nu1[edx][0] = nu1_[j_][0]; 
    nu1[edx][1] = nu1_[j_][1]; 
    nu1[edx][2] = nu1_[j_][2]; 
    nu1[edx][3] = nu1_[j_][3]; 

    nu2[edx][0] = nu2_[j_][0]; 
    nu2[edx][1] = nu2_[j_][1]; 
    nu2[edx][2] = nu2_[j_][2]; 
    nu2[edx][3] = nu2_[j_][3]; 
    cur_sol[edx] = solx;  
}

#endif
