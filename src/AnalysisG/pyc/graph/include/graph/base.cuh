#ifndef CUGRAPH_BASE_H
#define CUGRAPH_BASE_H
#include <utils/atomic.cuh>

template <typename scalar_t>
__global__ void _prediction_topology(
              torch::PackedTensorAccessor64<long, 3, torch::RestrictPtrTraits> pairs, 
        const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edge_index, 
        const torch::PackedTensorAccessor64<long, 1, torch::RestrictPtrTraits> pred,
        const unsigned int dx_lx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= dx_lx){return;}
    long prd = pred[idx]; 
    long src = edge_index[0][idx]; 
    long dst = edge_index[1][idx]; 
    pairs[prd][src][dst] = dst; 
    if (src != dst){return;}
    if (src < 0 || dst < 0){return;}
    for (size_t x(0); x < pairs.size({0}); ++x){pairs[x][src][dst] = src;}
}

template <typename scalar_t>
__global__ void _edge_summing(
              torch::PackedTensorAccessor64<long    , 3, torch::RestrictPtrTraits> pairs, 
              torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> pmu, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> pmi, 
        const unsigned int pred_lx, const unsigned int node_lx, const unsigned int node_fx
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z; 
    if (idx >= node_lx || idy >= node_fx || idz >= pred_lx){return;}

    double sx = 0; 
    for (size_t x(0); x < node_lx; ++x){sx += (pairs[idz][idx][x] >= 0) * pmi[x][idy];}
    pmu[idz][idx][idy] = sx; 
}


template <typename scalar_t, size_t size_x, size_t size_y, size_t size_z> 
__global__ void _fast_unique(
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> features, 
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out_fea, 
              torch::PackedTensorAccessor64<    long, 1, torch::RestrictPtrTraits> maxi_o, 
        const torch::PackedTensorAccessor64<    long, 2, torch::RestrictPtrTraits> cluster_map, 
              torch::PackedTensorAccessor64<    long, 2, torch::RestrictPtrTraits> out_map, 
        const unsigned int dim_i, const unsigned int dim_j, 
        const unsigned int dim_l, const unsigned int dim_e
){
    __shared__ long   msk[size_x][size_y];
    __shared__ long   skx[size_x][size_y];
    __shared__ double fea[size_x][size_z]; 

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int ix  = threadIdx.x;
    const unsigned int idy = threadIdx.y;
    const unsigned int dim_f = (dim_l < size_z) ? dim_l : size_z; 
    const unsigned int smx = size_x; 
    const bool blk = idx < dim_i; 

    if (blk && idy <  dim_j){msk[threadIdx.x][idy] = cluster_map[idx][idy];}
    if (blk && idy >= dim_j){msk[threadIdx.x][idy] = -1;}
    skx[ix][idy] = -1;

    __syncthreads();
    if (blk){
        long xk = msk[ix][idy]; 
        for (size_t y(0); y < size_y; ++y){
            long vl = msk[ix][y]; 
            if (vl < 0 || vl != xk){continue;}
            if (idy != y){msk[ix][idy] = -1; break;}
            skx[ix][idy] = vl; 
            msk[ix][idy] = -1;
            break; 
        }
    }
    __syncthreads(); 
    long pos = 0;  // count the relative offset
    long mxp = 0;  // count the global offset 
    long lx = skx[ix][idy]; 
    for (size_t y(0); y < size_y * blk; ++y){
        pos += (skx[ix][y] > -1 && lx > -1 && y < idy);
        mxp += (skx[ix][y] > -1); 
    }
    if (lx > -1){msk[ix][pos] = lx;}
    __syncthreads(); 
    skx[ix][idy] = -1; 
    __syncthreads();

    double fl = 0; 
    for (int y(0); y < mxp; ++y){
        long xl  = msk[ix][y]; 
        if (idy >= dim_f){continue;}
        if (xl < 0 || xl >= dim_e){continue;}
        // -------------------------------------------- 
        // if requested node is out of the current block scope:
        // fetch from global memory ~ slower but better than nothing, 
        // if within block, we use cached memory.
        //  -------------------------------------------

        long lxi = xl % smx; 
        double dl = 0; 
        if (skx[lxi][idy] != xl){
            dl = features[xl][idy];
            fea[lxi][idy] = dl;
            skx[lxi][idy] = xl;
        }
        else {dl = fea[lxi][idy];}

        fl += dl; 
    }
    __syncthreads(); 
    if (idy < size_z){fea[ix][idy] = fl;}
    __syncthreads(); 

    if (blk && idy < size_z){out_fea[idx][idy] = fea[ix][idy];}
    if (blk && idy < mxp   ){out_map[idx][idy] = msk[ix][idy];}
    if (blk && idy == dim_f){maxi_o[idx] = mxp;}
    
}


template <typename scalar_t>
__global__ void _unique_sum(
              torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> out, 
        const torch::PackedTensorAccessor64<long    , 2, torch::RestrictPtrTraits> cluster_map, 
        const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> features, 
        const unsigned int dim_i, const unsigned int dim_j, const unsigned int dim_k
){
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y; 
    if (idx >= dim_i || idy >= dim_k){return;}

    scalar_t sx = 0; 
    for (unsigned int i(0); i < dim_j; ++i){
        const long tx = cluster_map[idx][i]; 
        if (tx < 0 || tx > dim_j){continue;}
        sx += features[ tx ][idy];   
    }
    out[idx][idy] = sx; 
}


template <typename scalar_t, size_t size_x, size_t size_y> 
__global__ void _cycle_build(
        const torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> clu_map, 
              torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> cyl_map, 
              torch::PackedTensorAccessor64<long, 2, torch::RestrictPtrTraits> edges, 
              torch::PackedTensorAccessor64<bool, 1, torch::RestrictPtrTraits> msked, 
        const unsigned int dim_i, const unsigned int dim_j
){
    __shared__ long _cycle[size_x][size_y];
    __shared__ long _sink[size_x];         

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int n_tiles = (dim_i + size_x - 1) / size_x;

    long root = -1;
    for (int x(0); x < dim_j * (idx < dim_i); ++x){
        root = clu_map[idx][x]; 
        if (root < 0){continue;}
        break; 
    }

    int  cl = 0;
    long cyc[size_y];
    for (int x(0); x < dim_j * (idx < dim_i && root > -1); ++x){
        long val = clu_map[idx][x];
        if (val < 0){break;}
        bool found = false;
        for (int k(0); k < cl; ++k){
            if (cyc[k] != val){continue;}
            found = true; break;
        }
        if (found){continue;}
        if (cl >= size_y){continue;}
        cyc[cl++] = val;
    }

    for (unsigned int tile(0); tile < n_tiles; ++tile) {
        unsigned int row = tile * size_x + tid; 
        if (row < dim_i){
            for (int x(0); x < dim_j; ++x){_cycle[tid][x] = clu_map[row][x];}
            for (int x(dim_j); x < size_y; ++x){_cycle[tid][x] = -1;}
            long s = -1;
            for (int x = dim_j - 1; x >= 0; --x) {
                if (_cycle[tid][x] < 0){continue;}
                s = _cycle[tid][x]; break;
            }
            _sink[tid] = s;
        } 
        else {
            for (int x(0); x < size_y; ++x){_cycle[tid][x] = -1;}
            _sink[tid] = -1;
        }
        __syncthreads(); 

        for (unsigned int i(0); i < size_x * (idx < dim_i && root > -1); ++i){
            if (_sink[i] != root){continue;}
            for (int x(0); x < size_y; ++x){
                long val = _cycle[i][x];
                if (val < 0){break;}
                bool found = false;
                for (int k(0); k < cl; ++k){
                    if (cyc[k] != val){continue;}
                    found = true; break;
                }
                if (found){continue;}
                if (cl >= size_y){continue;}
                cyc[cl++] = val;
            }
        }
        __syncthreads();
    }
    for (int x(0); x < dim_j * (idx < dim_i); ++x){
        long val = (x < cl) ? cyc[x] : -1;
        if (val < 0){continue;}
        edges[0][idx * dim_j + x] = idx; 
        edges[1][idx * dim_j + x] = val; 
        msked[idx * dim_j + x] = true; 
        cyl_map[idx][x] = val; 
    }
}


#endif
