#include <nusol/base.cuh>
#include <nusol/device.cuh>
#include <cutils/utils.cuh>
#include <operators/operators.cuh>

template <size_t size_x>
__global__ void _swapAB(
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> inv_A_dot_B,
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> A,
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> B,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> detA, 
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> detB
){

    __shared__ double A_[size_x][3][3]; 
    __shared__ double B_[size_x][3][3]; 

    __shared__ double _cofA[size_x][3][3];  
    __shared__ double _detA[size_x][3][3];
    __shared__ double _InvA[size_x][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idx = threadIdx.x; 
    const unsigned int idy = threadIdx.y; 
    const unsigned int idz = threadIdx.z; 
    if (_idx >= detB.size({0})){return;}

    A_[idx][idy][idz] = A[_idx][idy][idz]; 
    B_[idx][idy][idz] = B[_idx][idy][idz]; 

    // ----- swap if abs(det(B)) > abs(det(A)) -------- //
    bool swp = abs(detB[_idx][0]) > abs(detA[_idx][0]); 
    double a_ = (!swp)*A_[idx][idy][idz] + (swp)*B_[idx][idy][idz]; 
    double b_ = (!swp)*B_[idx][idy][idz] + (swp)*A_[idx][idy][idz];
    A_[idx][idy][idz] = a_;  
    B_[idx][idy][idz] = b_;  
    __syncthreads(); 

    // ----- compute the inverse of A -------- //
    double mx = _cofactor(A_[idx], idy, idz); 
    _cofA[idx][idz][idy] = mx;  // transpose cofactor matrix to get adjoint 
    _detA[idx][idy][idz] = mx * A_[idx][idy][idz]; 
    __syncthreads(); 

    double _dt = _detA[idx][0][0] + _detA[idx][0][1] + _detA[idx][0][2];
    _InvA[idx][idy][idz] = _cofA[idx][idy][idz]*_div(&_dt); 
    __syncthreads(); 
    // --------------------------------------- //

    // -------- take the dot product of inv(A) and B --------- //
    inv_A_dot_B[_idx][idy][idz] = _dot(_InvA[idx], B_[idx], idy, idz, 3); 
    B[_idx][idy][idz] = B_[idx][idy][idz]; 
    A[_idx][idy][idz] = A_[idx][idy][idz]; 
} 

template <typename scalar_t>
__global__ void _factor_degen(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> real,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> imag,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> B,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> Lins,
        const double nulls
){
    __shared__ double G[3][3][3]; 
    __shared__ double g[3][3][3]; 
    __shared__ double coG[3][3][3]; 
    __shared__ double lines[3][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _ixd = threadIdx.y*9 + threadIdx.z; 
    const unsigned int _idt = (_ixd / 9)%3; 
    const unsigned int _idy = (_ixd / 3)%3; 
    const unsigned int _idz =  _ixd % 3; 
    if (abs(imag[_idx][_idt]) > 0.0){return;}

    G[_idt][_idy][_idz] = B[_idx][_idy][_idz] - real[_idx][_idt] * A[_idx][_idy][_idz]; 
    __syncthreads(); 

    bool c1 = (G[_idt][0][0] == G[_idt][1][1]) * (G[_idt][1][1] == 0); 
    if (c1){Lins[_idx][_idt][_idy][_idz] = _case1(G[_idt], _idy, _idz); return;}

    lines[_idt][_idy][_idz] = 0; 
    bool sw = abs(G[_idt][0][0]) > abs(G[_idt][1][1]); 
    g[_idt][_idy][_idz] = _case2(G[_idt], _idy, _idz, sw)*_div(G[_idt][!sw][!sw]);
    __syncthreads();

    coG[_idt][_idy][_idz] = _cofactor(g[_idt], _idy, _idz); 
    __syncthreads(); 

    double elx = 0; 
    if (-coG[_idt][2][2] <= nulls){ elx = _leqnulls(coG[_idt], g[_idt], _idy, _idz); }
    else { elx = _gnulls(coG[_idt], g[_idt], _idy, _idz); }

    if (_idz == 0){Lins[_idx][_idt][_idy][1 - !sw] = elx;}
    if (_idz == 1){Lins[_idx][_idt][_idy][!sw] = elx;}
    if (_idz == 2){Lins[_idx][_idt][_idy][2] = elx;}
}

template <typename scalar_t>
__global__ void _intersections(
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> real,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> imag,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> ellipse,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> lines,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_pts,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_dst,
        const double nulls
){
    __shared__ double _real[9][3][3]; 
    __shared__ double _imag[9][3][3];

    __shared__ double _elip[9][3][3]; 
    __shared__ double _line[9][3][3]; 
    __shared__ double _solx[9][3][3]; 
    __shared__ double _soly[9][3][3]; 
    __shared__ double _dist[9][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = threadIdx.y/9;
    const unsigned int _idz = threadIdx.y/3;
    const unsigned int _idt = threadIdx.y%3; 
    
    _real[_idz][_idt][threadIdx.z] = real[_idx][_idz][_idt][threadIdx.z]; 
    _imag[_idz][_idt][threadIdx.z] = imag[_idx][_idz][_idt][threadIdx.z]; 

    _line[_idz][_idt][threadIdx.z] = lines[_idx][_idy][_idz%3][threadIdx.z];
    _elip[_idz][_idt][threadIdx.z] = ellipse[_idx][_idt][threadIdx.z];
    __syncthreads();  

    double v1 = 0; 
    for (size_t x(0); x < 3; ++x){v1 += _line[_idz][_idt][x] * _real[_idz][_idt][x];}
    _soly[_idz][_idt][threadIdx.z] = _dot(_real[_idz], _elip[_idz], _idt, threadIdx.z, 3); 
    _solx[_idz][_idt][threadIdx.z] = _real[_idz][_idt][threadIdx.z]*_div(_real[_idz][_idt][2]); 
    __syncthreads(); 

    double v2 = 0; 
    for (size_t x(0); x < 3; ++x){v2 += _soly[_idz][_idt][x] * _real[_idz][_idt][x];}
    v2 = (v2 + v1) ? log10(v2*v2 + v1*v1) : 200;

    _dist[_idz][_idt][threadIdx.z] = (v2 < log10(nulls)) ? v2 : 200; 
    s_pts[_idx][_idz][_idt][threadIdx.z] = _dist[_idz][_idt][threadIdx.z]; 
    s_dst[_idx][_idz][_idt][threadIdx.z] = _solx[_idz][_idt][threadIdx.z]; 
}


template <typename scalar_t>
__global__ void _solsx(
        const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_pts,
        const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_dst,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> sols,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> solx,
        torch::PackedTensorAccessor64<long    , 4, torch::RestrictPtrTraits> idxs
){
    __shared__ double _point[9][3][3]; 
    __shared__ double _lines[9][3][3]; 
    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    const unsigned int _id1 = _idy/3; 
    const unsigned int _id2 = _idy%3; 
    
    long id_ = idxs[_idx][_id1][_id2][_idz]; 
    _point[_id1][_id2][_idz] = s_pts[_idx][_id1][id_][_idz]; 
    _lines[_id1][_id2][_idz] = s_dst[_idx][_id1][id_][_idz]; 
    __syncthreads(); 

    const unsigned dx_ = _id1*2 + _id2; 
    if (_id2 < 2){sols[_idx][dx_][_idz] = _lines[_id1][_id2][_idz];}
    if (_id2 < 2 && !threadIdx.z){solx[_idx][dx_][0] = _point[_id1][_id2][0];}
}


std::map<std::string, torch::Tensor> nusol_::Intersection(torch::Tensor* A, torch::Tensor* B, double nulls){
    const unsigned int dx = A -> size({0}); 
    const unsigned int thx = (dx >= 64) ? 64 : dx; 
    const dim3 thd  = dim3(thx, 3, 3);
    const dim3 thdX = dim3(1, 9, 9);
    const dim3 thdY = dim3(1, 27, 3); 

    const dim3 blk  = blk_(dx, thx, 3, 3, 3, 3); 
    const dim3 blkX = blk_(dx, 1, 9, 9, 9, 9); 
    const dim3 blkY = blk_(dx, 1, 27, 27, 3, 3); 

    torch::Tensor inv_A_dot_B = torch::zeros_like(*A); 
    torch::Tensor lines = torch::zeros({dx, 3, 3, 3}, MakeOp(A)); 
    torch::Tensor s_pts = torch::zeros({dx, 9, 3, 3}, MakeOp(A)); 
    torch::Tensor s_dst = torch::zeros({dx, 9, 3, 3}, MakeOp(A)); 

    torch::Tensor a_ = operators_::Determinant(A); 
    torch::Tensor b_ = operators_::Determinant(B);
    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "swp", [&]{
        _swapAB<64><<<blk, thd>>>(
          inv_A_dot_B.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                 A -> packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                 B -> packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                   a_.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                   b_.packed_accessor64<double, 2, torch::RestrictPtrTraits>()); 
    });

    //std::tuple<torch::Tensor, torch::Tensor> eig = operators_::Eigenvalue(&inv_A_dot_B); 
    torch::Tensor eig = torch::linalg::eigvals(inv_A_dot_B); 
    torch::Tensor real = torch::real(eig).to(A -> scalar_type()); //std::get<0>(eig); 
    torch::Tensor imag = torch::imag(eig).to(A -> scalar_type()); //std::get<1>(eig); 
    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "B-e*A", [&]{
        _factor_degen<scalar_t><<<blkX, thdX>>>(
                 real.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                 imag.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                 A -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 B -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                lines.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                nulls); 
    });

    std::vector<signed long> dim313 = {-1, 9, 1, 3}; 
    std::vector<signed long> dim133 = {-1, 1, 3, 3}; 
    torch::Tensor V = torch::cross(lines.view(dim313), A -> view(dim133), 3); 
    V = torch::transpose(V, 2, 3);
    V = std::get<1>(torch::linalg::eig(V)); 
    V = torch::transpose(V, 2, 3).view({-1, 9, 3, 3}); 
    real = torch::real(V);
    imag = torch::imag(V); 

    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "intersection", [&]{
        _intersections<scalar_t><<<blkY, thdY>>>(
                 real.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                 imag.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                 A -> packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                lines.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                s_pts.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                s_dst.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                nulls); 
    });

    torch::Tensor sols = torch::zeros({dx, 18, 3}, MakeOp(A)); 
    torch::Tensor solx = torch::zeros({dx, 18, 1}, MakeOp(A)); 
    torch::Tensor idx = std::get<1>(s_pts.sort(-2, false)); 
    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "sorted", [&]{
        _solsx<scalar_t><<<blkY, thdY>>>(
                s_pts.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                s_dst.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
                 sols.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 solx.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                  idx.packed_accessor64<long    , 4, torch::RestrictPtrTraits>()); 
    });
 
    std::map<std::string, torch::Tensor> out;
    out["solutions"] = sols; 
    out["distances"] = solx; 
    return out; 
}


