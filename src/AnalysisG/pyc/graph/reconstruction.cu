#include <physics/physics.cuh>
#include <graph/pagerank.cuh>
#include <utils/atomic.cuh>
#include <utils/utils.cuh>


template <typename scalar_t, size_t size_x>
__global__ void _find_unique(
    torch::PackedTensorAccessor64<long    , 1, torch::RestrictPtrTraits> node_index, 
    torch::PackedTensorAccessor64<long    , 3, torch::RestrictPtrTraits> count, 
    torch::PackedTensorAccessor64<double  , 2, torch::RestrictPtrTraits> pmc,

    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pgrc,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pgrk,
    torch::PackedTensorAccessor64<long    , 2, torch::RestrictPtrTraits> trk, 

    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> page_cluster,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> page_node,
    torch::PackedTensorAccessor64<double  , 3, torch::RestrictPtrTraits> pmc_out,
    const unsigned long num_n
){
    __shared__ bool     _found[size_x]; 
    __shared__ scalar_t _pcl[size_x];
    __shared__ scalar_t _pcn[size_x];
    __shared__ long     _crn[size_x]; 

    const unsigned int _idz = threadIdx.z; 
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z; 
   
    if (idy >= num_n || idz >= num_n){return;}
    long ix = count[idx][idy][idz]; 
    long in = node_index[idx]; 

    _pcl[_idz] = pgrc[idx][idz]; 
    _pcn[_idz] = pgrk[idx][idz];
    _crn[_idz] = ix;  

    long pos = -1; 
    for (size_t x(0); x < num_n; ++x){
        long ix_ = count[idx][x][idz]; 
        _found[_idz] = (ix_ == ix); 

        __syncthreads(); 
        bool found = true; 
        for (size_t f(0); f < num_n; ++f){found *= _found[f];}
        __syncthreads(); 
       
        if (!found){_found[_idz] = false; continue;}
        trk[idx][x] = x; 
        _found[_idz] = true;
        pos = x; 
        break; 
    }
    if (!_found[_idz] || ix < 0 || pos < 0){return;}
    
    double px(0), py(0), pz(0), e(0);  
    for (size_t x(0); x < num_n; ++x){ 
        long id = _crn[x] + in; 
        if (id < 0){continue;}

        px += pmc[id][0];  
        py += pmc[id][1];  
        pz += pmc[id][2];  
        e  += pmc[id][3];  
    }

    pmc_out[idx][pos][0] = px;  
    pmc_out[idx][pos][1] = py;  
    pmc_out[idx][pos][2] = pz;  
    pmc_out[idx][pos][3] = e;  

    page_cluster[idx][pos] = _pcl[_idz]; 
    page_node[idx][pos]    = _pcn[_idz]; 

}



std::map<std::string, torch::Tensor> graph_::page_rank_reconstruction(
            torch::Tensor* edge_index, torch::Tensor* edge_scores, torch::Tensor* pmc,  
            double alpha, double threshold, double norm_low, long timeout, int num_cls
){
    std::map<std::string, torch::Tensor> out = graph_::page_rank(edge_index, edge_scores, alpha, threshold, norm_low, timeout, num_cls);
    torch::Tensor node_count = out["nodes"]; 
    torch::Tensor page_rank  = out["pagerank"];  
    torch::Tensor page_clust = out["pagenode"];
    torch::Tensor node_index = out["node_index"];   

    const unsigned int num_nodes = node_count.size({1}); 
    const unsigned int num_event = node_count.size({0});
 
    const dim3 thdx  = dim3(1, 1, 64);
    const dim3 blkdx = blk_(num_event, 1, num_nodes, 1, num_nodes, 64); 
    torch::Tensor tkx = torch::zeros({num_event, num_nodes}, MakeOp(edge_index ))-1; 
    torch::Tensor prc = torch::zeros({num_event, num_nodes}, MakeOp(edge_scores))-1; 
    torch::Tensor prn = torch::zeros({num_event, num_nodes}, MakeOp(edge_scores))-1; 
    torch::Tensor pmx = torch::zeros({num_event, num_nodes, 4}, MakeOp(pmc));  

    AT_DISPATCH_ALL_TYPES(edge_scores -> scalar_type(), "get_max_node", [&]{
        _find_unique<scalar_t, 64><<<blkdx, thdx>>>(
                  node_index.packed_accessor64<long    , 1, torch::RestrictPtrTraits>(),
                  node_count.packed_accessor64<long    , 3, torch::RestrictPtrTraits>(),
                      pmc -> packed_accessor64<double  , 2, torch::RestrictPtrTraits>(),

                  page_clust.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                   page_rank.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                         tkx.packed_accessor64<long    , 2, torch::RestrictPtrTraits>(),

                         prc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                         prn.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                         pmx.packed_accessor64<double  , 3, torch::RestrictPtrTraits>(),
                        num_nodes
            ); 
    }); 

    torch::Tensor px = pmx.view({-1, 4});
    out["unique-pmc"] = pmx; 
    out["page-cluster"] = prc; 
    out["page-nodes"] = prn; 
    out["page-mass"] = physics_::M(&px).view({num_event, num_nodes}); 
    return out; 

}


