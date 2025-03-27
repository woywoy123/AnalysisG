#include <graph/pagerank.cuh>
#include <utils/atomic.cuh>
#include <utils/utils.cuh>

template <size_t size_x>
__global__ void _get_max_node(
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_inx, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> max_node, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> remap, 
          const int el, const long mxn
){
    __shared__ long _edge[size_x]; 

    const unsigned int _idx = threadIdx.x; 
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const bool skp = idx >= el; 

    long src = -2; 
    long dst = -2;  
    long mx_nodes = 0; 
    if (!skp){
        src = edge_inx[0][idx];
        dst = edge_inx[1][idx]; 
    }

    long lx = _idx; 
    for (size_t x(0); x < el; ++x){ // mx_nodes += (src == edge_inx[0][x]);}
        long mx = x % size_x; 
        if (!mx){
            __syncthreads();
            _edge[_idx] = (lx < el) ? edge_inx[0][lx] : -1; 
            lx += size_x; 
            __syncthreads(); 
        }
        mx_nodes += (src == _edge[mx]);
    }

    if (skp){return;}
    remap[src * mxn + dst] = long(idx); 
    max_node[idx] = mx_nodes; 
}

__global__ void _get_remapping(
    const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_inx, 
    const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> ev_node, 
          torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> idx_map, 
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
        ++xi;  
        if (xi != 1){continue;}
        if (edge_inx[0][lx] != edge_inx[1][lx]){continue;}
        num_batch[idx] = x;
        num_enode[idx] = ev_node[lx]; 
    }
}



template <typename scalar_t, size_t size_x>
__global__ void _page_rank(
    const torch::PackedTensorAccessor64<long    , 1, torch::RestrictPtrTraits> cu_xms, 
    const torch::PackedTensorAccessor64<long    , 1, torch::RestrictPtrTraits> cu_xme,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> edge_scores,
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pagerank,
          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pageclus,
          torch::PackedTensorAccessor64<long    , 3, torch::RestrictPtrTraits> count,
          torch::PackedTensorAccessor64<bool    , 1, torch::RestrictPtrTraits> edge_inx,
    const double alpha, 
    const double mlp_lim,
    const double threshold,
    const unsigned int mx_ev, 
    const unsigned int num_ev, 
    const unsigned int min_nodes, 
    const unsigned int mxn, 
    const long timeout
){
    __shared__ long _topx_mpx[size_x][size_x]; 
    __shared__ long _idx_remp[size_x][size_x]; 

    __shared__ scalar_t _topx_epx[size_x][size_x];  
    __shared__ scalar_t _topx_mij[size_x][size_x]; 

    __shared__ scalar_t _PgRank_o[size_x];
    __shared__ scalar_t _PgRank_t[size_x];

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= num_ev){return;}

    const unsigned int _idy    = threadIdx.y; 
    const unsigned int _idz    = threadIdx.z; 

    _idx_remp[_idy][_idz] = -1; 
    _topx_mpx[_idy][_idz] = -1; 

    _topx_epx[_idy][_idz] = 0; 
    _topx_mij[_idy][_idz] = 0; 
    _PgRank_o[_idy] = 0; 
    _PgRank_t[_idy] = 0; 

    unsigned int nodes = cu_xme[idx];  
    if (_idz >= nodes || _idy >= nodes){return;}
    const unsigned int ix = cu_xms[idx] + _idy*nodes + _idz%nodes; 
    if (nodes >= size_x){nodes = size_x;}

    const scalar_t s0 = edge_scores[0][ix]; 
    const scalar_t s1 = edge_scores[1][ix]; 
    scalar_t N = 1.0 / nodes; 
    if (s0 < s1 && s1 > mlp_lim){_topx_mpx[_idy][_idz] = _idz;}
    _idx_remp[_idy][_idz] = _topx_mpx[_idy][_idz];

    if (_idz != _idy){_topx_mij[_idz][_idy] = s1;}
    else {_PgRank_o[_idy] = s1 * N;}
    __syncthreads(); 

    // ---------------- prepare the matrix ------------------ //
    scalar_t sx = _sum(_topx_mij[_idy], nodes); 
    sx = alpha * ((sx) ? (1.0 / sx) : N) * _topx_mij[_idy][_idz]; 
    _topx_epx[_idz][_idy] = sx * _PgRank_o[_idy];  
    __syncthreads(); 
    // ------------------------------------------------------- //

    _topx_mij[_idz][_idy] = sx; 
    __syncthreads();

    N = (1 - alpha)*N; 
    sx = _topx_mij[_idy][_idz]; 

    // ----------------- Start the Main Algo ----------------- //
    scalar_t err = 10; 
    for (size_t x(0); x < timeout; ++x){
        scalar_t pr = _PgRank_o[_idy]; 
        _PgRank_t[_idy] = _sum(_topx_epx[_idy], nodes) + N;
        __syncthreads();

        // ----------------- Update Step --------------- //
        scalar_t sf = 1.0 / _sum(_PgRank_t, nodes); 
        _PgRank_o[_idy] = _PgRank_t[_idy]*sf; 
        _topx_mij[_idz][_idy] = abs(pr - _PgRank_o[_idy]); 
        _topx_epx[_idy][_idz] = sx * _PgRank_t[_idz] * sf; 
        __syncthreads(); 

        err = _sum(_topx_mij[_idz], nodes); 
        if (err > threshold){continue;}
        break; 
    }

    // -------------- perform clustering ----------- //
    for (int x(0); x < nodes * (_topx_mpx[_idy][_idz] != -1); ++x){
        long k = _topx_mpx[_idz][x]; 
        if (k == -1 || _idx_remp[_idy][k] > -1 || _topx_mpx[_idz][k] == -1){continue;}
        _idx_remp[_idy][k] = k; 
    }

    __syncthreads(); 
    if (err < threshold){
        _PgRank_t[_idy] = _sum(_topx_epx[_idy], nodes); 
        scalar_t norm = _sum(_PgRank_t, nodes); 
        if (!norm){_PgRank_o[_idy] = _PgRank_t[_idy];}
        else {_PgRank_o[_idy] = _PgRank_t[_idy] / norm;}
    }
    pagerank[idx][_idy] = _PgRank_o[_idy]; 
    if (!_PgRank_o[_idy]){return;}

    long n = 0; sx = 0; 
    for (size_t x(0); x < nodes; ++x){
        if (_idx_remp[_idy][x] == -1){continue;}
        sx += _PgRank_o[x]; ++n;
    }
    if (n < min_nodes){return;}
    edge_inx[ix] = _idx_remp[_idy][_idz] > -1; 
    count[idx][_idy][_idz] = _idx_remp[_idy][_idz];
    pageclus[idx][_idy] = sx; 
}

std::map<std::string, torch::Tensor> graph_::page_rank(
    torch::Tensor* edge_index, torch::Tensor* edge_scores, 
    double alpha, double threshold, double norm_low, long timeout, int num_cls
){
    const int iel = edge_index -> size({1}); 
    const long mxn = torch::max(edge_index -> index({0})).item<long>()+1; 
    torch::Tensor ev_node = -1 * torch::ones({iel      }, MakeOp(edge_index));
    torch::Tensor mx_remp = -1 * torch::ones({mxn * mxn}, MakeOp(edge_index));

    const dim3 thdx  = dim3(128);
    const dim3 blkdx = blk_(iel, 128); 

    AT_DISPATCH_ALL_TYPES(edge_index -> scalar_type(), "get_max_node", [&]{
        _get_max_node<128><<<blkdx, thdx>>>(
              edge_index -> packed_accessor64<long, 2, torch::RestrictPtrTraits>(),
                    ev_node.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                    mx_remp.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                    iel, mxn
            ); 
    }); 
       
    const long mx_ev = torch::max(ev_node).item<long>(); 
    torch::Tensor num_batch = torch::zeros({mxn}, MakeOp(edge_index))-1; 
    torch::Tensor num_enode = torch::zeros({mxn}, MakeOp(edge_index))-1;

    const dim3 thdnx = dim3(1024);
    const dim3 blknx = blk_(mxn, 1024); 

    AT_DISPATCH_ALL_TYPES(edge_index -> scalar_type(), "remapping", [&]{
        _get_remapping<<<blknx, thdnx>>>(
              edge_index -> packed_accessor64<long, 2, torch::RestrictPtrTraits>(),
                    ev_node.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                    mx_remp.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                  num_batch.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                  num_enode.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                    mxn, mx_ev
            ); 
    }); 

    torch::Tensor idx = num_batch > -1; 
    const unsigned int num_evnt = idx.sum({-1}).item<long>();
    torch::Tensor cu_em = num_enode.index({idx});
    torch::Tensor cu_xm = torch::cat({torch::zeros({1}, MakeOp(edge_index)), (cu_em * cu_em).cumsum({-1})}, {0}); 
    

    const dim3 thpr = dim3(1, 32, 32);
    const dim3 blpr = blk_(num_evnt, 1, 32, 32, 32, 32); 
    torch::Tensor page_rank  = torch::zeros({num_evnt, mx_ev       }, MakeOp(edge_scores)); 
    torch::Tensor page_clust = torch::zeros({num_evnt, mx_ev       }, MakeOp(edge_scores)); 
    torch::Tensor node_count = torch::zeros({num_evnt, mx_ev, mx_ev}, MakeOp(edge_index ))-1; 
    torch::Tensor edge_inx   = torch::zeros({iel                   }, MakeOp(edge_index )) > 0; 

    AT_DISPATCH_ALL_TYPES(edge_scores -> scalar_type(), "PageRank", [&]{
        _page_rank<scalar_t, 32><<<blpr, thpr>>>(
                    cu_xm.packed_accessor64<long    , 1, torch::RestrictPtrTraits>(),
                    cu_em.packed_accessor64<long    , 1, torch::RestrictPtrTraits>(), 
           edge_scores -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                page_rank.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
               page_clust.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
               node_count.packed_accessor64<long    , 3, torch::RestrictPtrTraits>(),
                 edge_inx.packed_accessor64<bool    , 1, torch::RestrictPtrTraits>(),
                    alpha, norm_low, threshold, mx_ev, num_evnt, num_cls, mxn, timeout
        ); 
    }); 
   
    std::map<std::string, torch::Tensor> out; 
    out["nodes"] = node_count; 
    out["pagerank"] = page_rank; 
    out["pagenode"] = page_clust; 
    out["edge_mask"] = edge_inx; 
    out["node_index"] = torch::cat({torch::zeros({1}, MakeOp(edge_index)), cu_em}, {0}); 
    return out; 
}

