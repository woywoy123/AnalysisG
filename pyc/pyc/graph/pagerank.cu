#include <graph/pagerank.cuh>
#include <utils/atomic.cuh>
#include <utils/utils.cuh>

template <size_t size_x>
__global__ void _get_max_node(
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_inx, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> max_node, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> remap, 
          const unsigned int el, const long mxn
){
    __shared__ long _edge[size_x]; 

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= el){return;}
    long src = edge_inx[0][idx];
    long dst = edge_inx[1][idx]; 
    //_edge[threadIdx.x] = dst;     
    //__syncthreads(); 

    size_t l(0);
    long mx_nodes = 0; 
    for (size_t x(0); x < el; ++x, ++l){ mx_nodes += (src == edge_inx[0][x]);}
        //if (l < size_x){mx_nodes += (src == _edge[l]); continue;}
        //__syncthreads(); 
        //_edge[threadIdx.x] = edge_inx[1][idx + size_x];
        //__syncthreads(); 
        //l = 0; --x; 
    //}

    remap[src * mxn + dst] = long(idx); 
    max_node[idx] = mx_nodes; 
}

__global__ void _get_remapping(
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_inx, 
    const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> ev_node, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> idx_map, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> idx_remap, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> num_batch, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> num_enode, 
          const long mxn, const long mxn_ev
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= mxn){return;}

    int xi = 0; 
    for (size_t x(0); x < mxn; ++x){
        unsigned int idl = idx * mxn + x; 
        long lx = idx_map[idl];
        if (lx < 0){continue;}
        idx_remap[idx*mxn_ev + xi] = lx;  
        ++xi;  
        if (xi != 1){continue;}
        if (edge_inx[0][lx] != edge_inx[1][lx]){continue;}
        num_batch[idx] = x;
        num_enode[idx] = ev_node[lx]; 
    }
}



template <size_t size_x>
__global__ void _page_rank(
    const torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> cu_xms, 
    const torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> cu_xme,
    const torch::PackedTensorAccessor64<long  , 1, torch::RestrictPtrTraits> idx_rmp,
    const torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> edge_scores,
          torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> pagerank,
          torch::PackedTensorAccessor64<long  , 3, torch::RestrictPtrTraits> counts,
    const double alpha, 
    const double mlp_lim,
    const double threshold,
    const unsigned int mx_ev, 
    const unsigned int num_ev, 
    const unsigned int min_nodes, 
    const unsigned int mxn, 
    const long timeout
){
    //__shared__ long   _idx_remp[size_x][size_x]; 
    //__shared__ long   _topx_mpx[size_x][size_x]; 

    __shared__ double _topx_epx[size_x][size_x];  
    __shared__ double _topx_mij[size_x][size_x]; 

    __shared__ double _PgRank_o[size_x];
    __shared__ double _PgRank_t[size_x];

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= num_ev){return;}

    const unsigned int _idy = threadIdx.y; 
    const unsigned int _idz = threadIdx.z; 
    const unsigned int nodes = cu_xme[idx];  
    const unsigned int exe   = mx_ev*nodes; 

    //_idx_remp[_idy][_idz] = 0; 
    //_topx_mpx[_idy][_idz] = 0; 
    _topx_epx[_idy][_idz] = 0; 
    _topx_mij[_idy][_idz] = 0; 
    _PgRank_o[_idy] = 0; 
    _PgRank_t[_idy] = 0; 

    const unsigned int idk = threadIdx.y * mx_ev + threadIdx.z; 
    if (_idz >= mx_ev || _idy >= nodes){return;}

    long  ix = idx_rmp[idx * mx_ev * nodes + idk];
    counts[idx][_idy][_idz] = long(ix);
    return; 
    if (ix == -1){return;}
    double N = 1.0 / nodes; 

    const double s0 = edge_scores[0][ix]; 
    const double s1 = edge_scores[1][ix]; 
    const long sl   = (s0 < s1) ? long(_idz) : long(-1); 
    //const bool skl  = s1 < mlp_lim; 

    //_topx_mpx[_idy][_idz] = sl; 
    if (_idz != _idy){_topx_mij[_idy][_idz] = s1;}
    else {_PgRank_o[_idy] = s1 * N;}
    __syncthreads(); 

    // ---------------- prepare the matrix ------------------ //
    double sx = 0; 
    for (size_t x(0); x < nodes; ++x){sx += _topx_mij[x][_idz];}
    sx = _topx_mij[_idy][_idz] * alpha * ((sx) ? (1.0 / sx) : N); 
    // ------------------------------------------------------- //

    N = (1 - alpha)*N; 
    // ----------------- Start the Main Algo ----------------- //
    for (size_t x(0); x < timeout; ++x){
        double pr_i = _PgRank_o[_idy]; 
        _topx_epx[_idy][_idz] = sx * _PgRank_o[_idz];  
        __syncthreads(); 

        _topx_mij[_idy][_idz] = 0; 
        _PgRank_t[_idy] = _sum(_topx_epx[_idy], nodes) + N;
        __syncthreads();

        // ----------------- Update Step --------------- //
        double pr = _PgRank_t[_idy]  / _sum(_PgRank_t, nodes); 
        _topx_mij[_idz][_idy] = abs(pr - pr_i); 
        _PgRank_o[_idy] = pr; 
        __syncthreads(); 

        if (_sum(_topx_mij[_idz], nodes) > threshold){continue;}

        _topx_epx[_idy][_idz] = sx * _PgRank_o[_idz];  
        __syncthreads();

        _PgRank_t[_idy] = _sum(_topx_epx[_idy], nodes); 
        __syncthreads(); 

        double norm = _sum(_PgRank_t, nodes); 
        if (!norm){_PgRank_o[_idy] = _PgRank_t[_idy];}
        else {_PgRank_o[_idy] = _PgRank_t[_idy] / norm;}
        //_idx_remp[_idy][_idz] = -1; 
        _topx_mij[_idy][_idz] = sx; 
        break; 
    }

    if (_idz){return;}
    pagerank[idx][_idy] = nodes; //_PgRank_o[_idy];
    return; 

    //long n = _topx_mpx[_idy][_idz]; 
    //if (n == -1 || _topx_mij[_idy][_idz] < mlp_lim){return;}
    //_idx_remp[_idy][_idz] = _idz; 
    //__syncthreads(); 

    //if (_idz){return;}
    //for (size_t x(0); x < nodes; ++x){
    //    long k = _topx_mpx[n][x]; 
    //    if (k == -1 || _topx_mpx[n][k] == -1 || _idx_remp[_idy][k] > -1 ){continue;}
    //    _idx_remp[_idy][k] = k; 
    //    x = 0; n = k; 
    //}

    //n = 0; sx = 0; 
    //for (size_t x(0); x < nodes; ++x){
    //    if (_idx_remp[_idy][x] == -1){continue;}
    //    _idx_remp[_idy][n] = _idx_remp[_idy][x]; 
    //    sx += _PgRank_o[x]; 
    //    ++n;
    //}
    //if (n < min_nodes){return;}
    //for (size_t x(0); x < n; ++x){counts[idx][_idy][x] = _idx_remp[_idy][x];}
    //pagerank[idx][_idy] = sx;  
}


std::map<std::string, torch::Tensor> graph_::page_rank(
    torch::Tensor* edge_index, torch::Tensor* edge_scores, 
    double alpha, double threshold, double norm_low, long timeout
){
    const unsigned int iel = edge_index -> size({1}); 
    const long mxn = torch::max(edge_index -> index({0})).item<long>()+1; 

    torch::Tensor ev_node = torch::zeros({iel      }, MakeOp(edge_index))-1;
    torch::Tensor mx_remp = torch::zeros({mxn * mxn}, MakeOp(edge_index))-1;

    const dim3 thdx  = dim3(1024);
    const dim3 blkdx = blk_(iel, 1024); 

    AT_DISPATCH_ALL_TYPES(edge_index -> scalar_type(), "get_max_node", [&]{
        _get_max_node<1024><<<blkdx, thdx>>>(
              edge_index -> packed_accessor64<long, 2, torch::RestrictPtrTraits>(),
                    ev_node.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                    mx_remp.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                    iel, mxn
            ); 
    }); 
        
    const long mx_ev = torch::max(ev_node).item<long>(); 
    torch::Tensor num_batch = torch::zeros({mxn        }, MakeOp(edge_index))-1; 
    torch::Tensor num_enode = torch::zeros({mxn        }, MakeOp(edge_index))-1;
    torch::Tensor idx_remap = torch::zeros({mxn * mx_ev}, MakeOp(edge_index))-1; 

    const dim3 thdnx = dim3(1024);
    const dim3 blknx = blk_(mxn, 1024); 

    AT_DISPATCH_ALL_TYPES(edge_index -> scalar_type(), "remapping", [&]{
        _get_remapping<<<blknx, thdnx>>>(
              edge_index -> packed_accessor64<long, 2, torch::RestrictPtrTraits>(),
                    ev_node.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                    mx_remp.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                  idx_remap.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                  num_batch.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                  num_enode.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                    mxn, mx_ev
            ); 
    }); 
    
    torch::Tensor idx = num_batch > -1; 
    const unsigned int num_evnt = idx.sum({-1}).item<long>();
    torch::Tensor cu_xm = num_batch.index({idx}); 
    torch::Tensor cu_em = num_enode.index({idx});

    const dim3 thpr = dim3(1, 32, 32);
    const dim3 blpr = blk_(num_evnt, 1, 32, 32, 32, 32); 

    torch::Tensor page_ranked = torch::zeros({num_evnt, mx_ev       }, MakeOp(edge_scores)); 
    torch::Tensor node_counts = torch::zeros({num_evnt, mx_ev, mx_ev}, MakeOp(edge_index))-1; 
    AT_DISPATCH_ALL_TYPES(edge_scores -> scalar_type(), "PageRank", [&]{
        _page_rank<32><<<blpr, thpr>>>(
                    cu_xm.packed_accessor64<long  , 1, torch::RestrictPtrTraits>(),
                    cu_em.packed_accessor64<long  , 1, torch::RestrictPtrTraits>(), 
                idx_remap.packed_accessor64<long  , 1, torch::RestrictPtrTraits>(),
           edge_scores -> packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
              page_ranked.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
              node_counts.packed_accessor64<long  , 3, torch::RestrictPtrTraits>(),
                    alpha, norm_low, threshold, mx_ev, num_evnt, 2, mxn, timeout
        ); 
    }); 
    
    std::map<std::string, torch::Tensor> out; 
    out["mxn"] = ev_node; 
    out["mx_remp"] = mx_remp; 
    out["remap"] = idx_remap; 
    out["num_e"] = num_enode;
    out["num_b"] = num_batch; 
    out["cu_xm"] = cu_xm; 
    out["cu_em"] = cu_em; 
    out["nodes"] = node_counts; 
    out["pagerank"] = page_ranked; 
    return out; 
}

