#include <nusol/base.cuh>
#include <nusol/device.cuh>
#include <cutils/utils.cuh>
#include <operators/operators.cuh>

#define _th_inter 8


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

template <typename scalar_t, size_t size_x>
__global__ void _factor_degen(
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> real,
        torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> imag,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> A,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> B,
        torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> Lins,
        const double nulls
){
    __shared__ double G[size_x][3][3][3]; 
    __shared__ double g[size_x][3][3][3]; 
    __shared__ double coG[size_x][3][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _ixd = threadIdx.y*9 + threadIdx.z; 
    const unsigned int _idt = (_ixd / 9)%3; 
    const unsigned int _idy = (_ixd / 3)%3; 
    const unsigned int _idz =  _ixd % 3; 
    const unsigned int idx  = threadIdx.x; 
    if (_idx >= imag.size({0})){return;}
    if (abs(imag[_idx][_idt]) > 0.0){return;}

    G[idx][_idt][_idy][_idz] = B[_idx][_idy][_idz] - real[_idx][_idt] * A[_idx][_idy][_idz]; 
    __syncthreads(); 

    bool c1 = (G[idx][_idt][0][0] == G[idx][_idt][1][1]) * (G[idx][_idt][1][1] == 0); 
    if (c1){Lins[_idx][_idt][_idy][_idz] = _case1(G[idx][_idt], _idy, _idz); return;}

    bool sw = abs(G[idx][_idt][0][0]) > abs(G[idx][_idt][1][1]); 
    g[idx][_idt][_idy][_idz] = _case2(G[idx][_idt], _idy, _idz, sw)*_div(G[idx][_idt][!sw][!sw]);
    __syncthreads();

    coG[idx][_idt][_idy][_idz] = _cofactor(g[idx][_idt], _idy, _idz); 
    __syncthreads(); 

    double elx = 0; 
    if (-coG[idx][_idt][2][2] <= nulls){ elx = _leqnulls(coG[idx][_idt], g[idx][_idt], _idy, _idz); }
    else { elx = _gnulls(coG[idx][_idt], g[idx][_idt], _idy, _idz); }

    if (_idz == 0){Lins[_idx][_idt][_idy][1 - !sw] = elx;}
    if (_idz == 1){Lins[_idx][_idt][_idy][!sw] = elx;}
    if (_idz == 2){Lins[_idx][_idt][_idy][2] = elx;}
}

template <typename scalar_t, size_t size_x>
__global__ void _intersections(
        torch::PackedTensorAccessor64<c10::complex<double>, 4, torch::RestrictPtrTraits> real,
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> ellipse,
        torch::PackedTensorAccessor64<double, 4, torch::RestrictPtrTraits> lines,
        torch::PackedTensorAccessor64<double, 3, torch::RestrictPtrTraits> s_pts,
        torch::PackedTensorAccessor64<double, 2, torch::RestrictPtrTraits> s_dst,
        const double nulls
){
    __shared__ double _real[size_x][9][3][3]; 
    __shared__ double _elip[size_x][9][3][3]; 
    __shared__ double _line[size_x][9][3][3]; 
    __shared__ double _solx[size_x][9][3][3]; 
    __shared__ double _ptsx[size_x][27][3]; 

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int idy = threadIdx.y;
    const unsigned int idz = threadIdx.z; 
    const unsigned int idt = idy / 3;
    const unsigned int idv = idy % 3;   
    const unsigned int id = 200 - idy;
    if (idx >= s_dst.size({0})){return;}

    c10::complex<double> vr = real[idx][idt][idv][idz]; 
    _real[threadIdx.x][idt][idv][idz] = vr.real(); 
    _line[threadIdx.x][idt][idv][idz] = lines[idx][(idy / 9)%3][idt%3][idz]; 
    _elip[threadIdx.x][idt][idv][idz] = ellipse[idx][idv][idz]; 
    _ptsx[threadIdx.x][idy][idz] = vr.imag() == 0; 
    __syncthreads(); 

    bool msk = true; 
    for (size_t z(0); z < 3; ++z){msk *= _ptsx[threadIdx.x][idy][z];}
    _solx[threadIdx.x][idt][idv][idz] = _dot(_real[threadIdx.x][idt], _elip[threadIdx.x][idt], idv, idz, 3); 
    double pts = _real[threadIdx.x][idt][idv][idz] * _div(_real[threadIdx.x][idt][idv][2]); 
    double v1  = _dot(_line[threadIdx.x][idt][idv], _real[threadIdx.x][idt][idv], 3);
    __syncthreads(); 

    double v2 = _dot(_solx[threadIdx.x][idt][idv], _real[threadIdx.x][idt][idv], 3); 
    double dist = v2*v2 + v1*v1; 
    msk *= dist != 0; 

    dist = log10(dist*msk + !msk) + (!msk)*id; 
    msk *= dist < log10(nulls); 
    dist = msk*dist + (!msk)*id; 

    _ptsx[threadIdx.x][idy][idz] = dist * (idz == 0); 
    __syncthreads(); 

    int pos = 0; 
    for (size_t y(0); y < 27; ++y){pos += (dist > _ptsx[threadIdx.x][y][0]);}
    _ptsx[threadIdx.x][pos][1] = dist; 
    __syncthreads(); 

    pos = 0; 
    for (size_t y(0); y < 27; ++y){pos += (dist > _ptsx[threadIdx.x][y][1]) * (0 != _ptsx[threadIdx.x][y][1]);}
    if (pos > 5){return;}
    s_dst[idx][pos] = dist;
    s_pts[idx][pos][idz] = pts*msk; 
}


template <typename scalar_t, size_t size_x>
__global__ void _solsx(
        const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_pts,
        const torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> s_dst,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> sols,
        torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> solx,
        torch::PackedTensorAccessor64<long    , 4, torch::RestrictPtrTraits> idxs
){
    __shared__ double _point[size_x][9][3][3]; 
    __shared__ double _lines[size_x][9][3][3]; 

    const unsigned int _idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const unsigned int _idy = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int _idz = blockIdx.z * blockDim.z + threadIdx.z; 
    const unsigned int _id1 = _idy/3; 
    const unsigned int _id2 = _idy%3; 
    if (_idx >= s_pts.size({0})){return;}

    long id_ = idxs[_idx][_id1][_id2][_idz]; 
    _point[threadIdx.x][_id1][_id2][_idz] = s_pts[_idx][_id1][id_][_idz]; 
    _lines[threadIdx.x][_id1][_id2][_idz] = s_dst[_idx][_id1][id_][_idz]; 
    __syncthreads(); 

    const unsigned dx_ = _id1*2 + _id2; 
    if (_id2 < 2){sols[_idx][dx_][_idz] = _lines[threadIdx.x][_id1][_id2][_idz];}
    if (_id2 < 2 && !threadIdx.z){solx[_idx][dx_][0] = _point[threadIdx.x][_id1][_id2][0];}
}


std::map<std::string, torch::Tensor> nusol_::Intersection(torch::Tensor* A, torch::Tensor* B, double nulls){
    const unsigned int dx = A -> size({0}); 
    const unsigned int thx = (dx >= _th_inter) ? _th_inter : dx; 
    const dim3 thd  = dim3(thx, 3, 3);
    const dim3 thdX = dim3(thx, 9, 9);
    const dim3 thdY = dim3(thx, 27, 3); 

    const dim3 blk  = blk_(dx, thx, 3, 3, 3, 3); 
    const dim3 blkX = blk_(dx, thx, 9, 9, 9, 9); 
    const dim3 blkY = blk_(dx, thx, 27, 27, 3, 3); 

    torch::Tensor inv_A_dot_B = torch::zeros_like(*A); 
    torch::Tensor lines = torch::zeros({dx, 3, 3, 3}, MakeOp(A)); 

    torch::Tensor a_ = operators_::Determinant(A); 
    torch::Tensor b_ = operators_::Determinant(B);
    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "swp", [&]{
        _swapAB<_th_inter><<<blk, thd>>>(
          inv_A_dot_B.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                 A -> packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                 B -> packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                   a_.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                   b_.packed_accessor64<double, 2, torch::RestrictPtrTraits>()); 
    });


    std::tuple<torch::Tensor, torch::Tensor> eigf = operators_::Eigenvalue(&inv_A_dot_B); 
    torch::Tensor real = std::get<0>(eigf); 
    torch::Tensor imag = std::get<1>(eigf); 
    torch::Tensor  msk = real.sum(-1) == 0 * imag.sum(-1) != 0; 
    if (msk.index({msk}).size({0}) != msk.size({0})){
        torch::Tensor eig = torch::linalg::eigvals(inv_A_dot_B.index({msk == false})); 
        real.index_put_({msk == false}, torch::real(eig).to(A -> scalar_type())); 
        imag.index_put_({msk == false}, torch::imag(eig).to(A -> scalar_type())); 
    }

    AT_DISPATCH_ALL_TYPES(A -> scalar_type(), "B-e*A", [&]{
        _factor_degen<scalar_t, _th_inter><<<blkX, thdX>>>(
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
    V = torch::transpose(V.view({-1, 3, 3}), 1, 2);
    V = std::get<1>(torch::linalg::eig(V)); 
    V = torch::transpose(V.view({dx, 9, 3, 3}), 2, 3); 

    torch::Tensor s_pts = torch::zeros({dx, 6, 3}, MakeOp(A)); 
    torch::Tensor s_dst = torch::zeros({dx, 6}, MakeOp(A)); 
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A -> scalar_type(), "intersection", [&]{
        _intersections<scalar_t, _th_inter><<<blkY, thdY>>>(
                    V.packed_accessor64<c10::complex<double>, 4, torch::RestrictPtrTraits>(),
                 A -> packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                lines.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),
                s_pts.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                s_dst.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                nulls); 
    });

    std::map<std::string, torch::Tensor> out;
    out["solutions"] = s_pts; 
    out["distances"] = s_dst; 
    return out; 
}


