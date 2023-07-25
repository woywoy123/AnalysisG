#include <transform/polar-cuda/polar.h>
#include <torch/torch.h>
#include <operators.h>
#include <physics.h>
#include <vector>
#include <map>

#ifndef NUSOL_CUDA_KERNEL_H
#define NUSOL_CUDA_KERNEL_H
#include "kernel.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static const dim3 BLOCKS(const unsigned int threads, const unsigned int len)
{
    const dim3 blocks( (len + threads -1) / threads ); 
    return blocks; 
}

static const dim3 BLOCKS(
    const unsigned int threads, const unsigned int len, const unsigned int dy)
{
    const dim3 blocks( (len + threads -1) / threads, dy); 
    return blocks; 
}

static const dim3 BLOCKS(
    const unsigned int threads, const unsigned int len, 
    const unsigned int dy, const unsigned int dz)
{
    const dim3 blocks( (len + threads -1) / threads, dy, dz); 
    return blocks; 
}

static const torch::TensorOptions _MakeOp(torch::Tensor v1)
{
	return torch::TensorOptions().dtype(v1.scalar_type()).device(v1.device()); 
}

torch::Tensor _Shape_Matrix(torch::Tensor inpt, std::vector<long> vec)
{
    const torch::TensorOptions op = _MakeOp(inpt); 
    const unsigned int len_i = inpt.size(0); 
    const unsigned int len_j = vec.size(); 
    torch::Tensor out = torch::zeros_like(inpt); 
    torch::Tensor vecT = torch::zeros({1, 1, len_j}, op).to(torch::kCPU); 
    for (unsigned int i(0); i < len_j; ++i){ vecT[0][0][i] += vec[i]; }
    vecT = vecT.to(inpt.device()); 

    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, len_i, len_j, len_j);
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]
    {
        _ShapeKernel<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                vecT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, len_j, len_j, false); 
    })); 
    return out; 
} 

torch::Tensor _Expand_Matrix(
        torch::Tensor inpt, torch::Tensor source)
{
    const unsigned int threads = 1024; 
    const unsigned int len_i = inpt.size(0);
    const unsigned int len_k = source.size(1); 
    source = source.view({source.size(0), len_k, -1}); 
    const unsigned int len_j = source.size(2); 
    const dim3 blk = BLOCKS(threads, len_i, len_k, len_j);
    torch::Tensor out = torch::zeros_like(inpt); 

    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]
    {
        _ShapeKernel<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                source.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, (len_k > inpt.size(1)) ? inpt.size(1) : len_k, len_j, true); 
    })); 
    return out; 
} 

torch::Tensor _Base_Matrix(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu)
{
    const unsigned int threads = 1024; 
    const unsigned int len_i = pmc_b.size(0); 
    const unsigned int len_m = masses_W_top_nu.size(0); 

    torch::Tensor beta2_b   = Physics::CUDA::Beta2(pmc_b); 
    torch::Tensor mass2_b   = Physics::CUDA::M2(pmc_b); 
    
    torch::Tensor beta2_mu  = Physics::CUDA::Beta2(pmc_mu);
    torch::Tensor mass2_mu  = Physics::CUDA::M2(pmc_mu); 
    
    torch::Tensor costheta  = Operators::CUDA::CosTheta(pmc_b, pmc_mu, 3);

    // [Z/Om, 0, x1 - p_mu], [ w * Z/Om, 0, y1 ], [0, Z, 0]
    torch::Tensor out = torch::zeros({len_i, 3, 3}, _MakeOp(costheta)); 
    const dim3 blk = BLOCKS(threads, len_i, 3, 3); 
    AT_DISPATCH_FLOATING_TYPES(costheta.scalar_type(), "BaseMatrix", ([&]
    {
        _H_Base<scalar_t><<< blk, threads>>>(
                    out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

                beta2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                mass2_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                  pmc_b.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 

               beta2_mu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               mass2_mu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                 pmc_mu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 

               costheta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
        masses_W_top_nu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               len_i, len_m); 
    })); 
    return out; 
} 

torch::Tensor _Base_Matrix_H(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses)
{
    torch::Tensor base  = _Base_Matrix(pmc_b, pmc_mu, masses);
    const torch::TensorOptions op = _MakeOp(base); 
    const unsigned int threads = 1024; 
    const unsigned int dim_i = pmc_b.size(0); 
    const dim3 blk  = BLOCKS(threads, dim_i, 6, 3); 
    const dim3 blk_ = BLOCKS(threads, dim_i, 3, 3); 

    torch::Tensor phi   = Transform::CUDA::Phi(pmc_mu); 
    torch::Tensor theta = Physics::CUDA::Theta(pmc_mu); 

    torch::Tensor Rx = _Expand_Matrix(base, pmc_b); 
    torch::Tensor Rz = torch::zeros({dim_i, 3, 3}, op); 
    torch::Tensor Ry = torch::zeros({dim_i, 3, 3}, op); 

    torch::Tensor RzT = torch::zeros({dim_i, 3, 3}, op); 
    torch::Tensor RyT = torch::zeros({dim_i, 3, 3}, op); 
    torch::Tensor RxT = torch::zeros({dim_i, 3, 3}, op);

    AT_DISPATCH_FLOATING_TYPES(phi.scalar_type(), "BaseMatrixH", ([&]
    {
        _Base_Matrix_H_Kernel<scalar_t><<< blk, threads >>>(
                Ry.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                Rz.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

               RyT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
               RzT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

               phi.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
             theta.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
               dim_i); 

        Rx = Operators::CUDA::Mul(Rz, Rx); 
        Rx = Operators::CUDA::Mul(Ry, Rx);

        _Base_Matrix_H_Kernel<scalar_t><<< blk_, threads >>>(
                Rx.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
               RxT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
               dim_i); 
        
    })); 
    Rx = Operators::CUDA::Mul(RyT, RxT); 
    Rx = Operators::CUDA::Mul(RzT, Rx); 
    return Operators::CUDA::Mul(Rx, base);
}

std::tuple<torch::Tensor, torch::Tensor> _DotMatrix(
        torch::Tensor MET_xy, torch::Tensor H, torch::Tensor Shape)
{
    const unsigned int dim_i = MET_xy.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3);

    MET_xy = _Expand_Matrix(H, MET_xy); 
    torch::Tensor   X = torch::zeros_like(H);
    torch::Tensor dNu = X.clone(); 
    AT_DISPATCH_FLOATING_TYPES(MET_xy.scalar_type(), "NuMatrix", ([&]
    {
        _V0_deltaK<scalar_t><<< blk, threads >>>(
                 X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
               dNu.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            MET_xy.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
             Shape.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                 H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 dim_i); 
    })); 

    return {Operators::CUDA::Mul(X, dNu), dNu}; 
}

std::tuple<torch::Tensor, torch::Tensor> _Intersection(
        torch::Tensor A, torch::Tensor B, const double null)
{
    const unsigned int dim_i = A.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3); 
    
    torch::Tensor det_A = Operators::CUDA::Determinant(A); 
    torch::Tensor det_B = Operators::CUDA::Determinant(B); 
    
    AT_DISPATCH_FLOATING_TYPES(det_A.scalar_type(), "Swap", ([&]
    {
        _SwapAB_K<scalar_t><<< blk, threads >>>(
                det_A.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                det_B.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                    A.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                    B.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                dim_i); 
    })); 

    torch::Tensor imag; 
    imag = Operators::CUDA::Inverse(A);
    imag = Operators::CUDA::Mul(imag, B);
    imag = torch::linalg::eigvals(imag); 
    
    const torch::TensorOptions op = _MakeOp(A); 
    const unsigned int dim_eig = imag.size(-1); 
    const dim3 blk_ = BLOCKS(threads, dim_i, 9, dim_eig); 

    torch::Tensor G   = torch::zeros({dim_i, dim_eig, 3, 3}, op); 
    torch::Tensor L   = torch::zeros({dim_i, dim_eig, 3, 3}, op);
    torch::Tensor O   = torch::zeros({dim_i, dim_eig, 3, 3}, op);
    torch::Tensor swp = torch::zeros({dim_i, dim_eig}, op.dtype(torch::kBool));  

    unsigned int size_swap = sizeof(unsigned int)*18; 
    unsigned int size_det  = sizeof(unsigned int)*12;

    // Oh the joy of C++.....
    unsigned int *sy, *sz, *dy, *dz;

    unsigned int _y[18] = {
        1, 1, 1, 0, 0, 0, 2, 2, 2, 
        0, 0, 0, 1, 1, 1, 2, 2, 2
    };

    unsigned int _z[18] = {
        1, 0, 2, 1, 0, 2, 1, 0, 2, 
        0, 1, 2, 0, 1, 2, 0, 1, 2
    }; 

    unsigned int _dy[12] = {
        1, 1, 2, 2, 
        0, 0, 2, 2, 
        0, 0, 1, 1
    }; 

    unsigned int _dz[12] = {
        1, 2, 1, 2, 
        0, 2, 0, 2, 
        0, 1, 0, 1
    }; 

    cudaMalloc(&sy, size_swap); 
    cudaMalloc(&sz, size_swap); 
    cudaMalloc(&dy, size_det); 
    cudaMalloc(&dz, size_det); 
    cudaMemcpy(sy, &_y , size_swap, cudaMemcpyHostToDevice); 
    cudaMemcpy(sz, &_z , size_swap, cudaMemcpyHostToDevice); 
    cudaMemcpy(dy, &_dy, size_det , cudaMemcpyHostToDevice); 
    cudaMemcpy(dz, &_dz, size_det , cudaMemcpyHostToDevice); 

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(imag.scalar_type(), "imaginary", ([&]
    {
        _imagineK<scalar_t><<< blk_, threads >>>(
            G.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
            A.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 
            B.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 
         imag.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dim_eig, dim_i); 
        
        _degenerateK<scalar_t><<< blk_, threads >>>(
            L.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
          swp.packed_accessor32<bool  , 2, torch::RestrictPtrTraits>(), 
            G.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
            dim_eig, dim_i, sy, sz); 

        _CoFactorK<scalar_t><<< blk_, threads >>>(
            G.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
            L.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
            dim_eig, dim_i, dy, dz);

        _FactorizeK<scalar_t><<< blk_, threads >>>(
            O.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),  
            L.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
            G.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),
            dim_eig, dim_i, null); 

        _SwapXY_K<scalar_t><<< blk_, threads >>>(
            G.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),
            O.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),  
          swp.packed_accessor32<bool  , 2, torch::RestrictPtrTraits>(), 
            dim_eig, dim_i); 
    }));  

    imag = torch::linalg_cross(G.view({-1, dim_eig*dim_eig, 1, 3}), A.view({-1, 1, 3, 3})); 
    imag = torch::transpose(imag, 2, 3);
    imag = std::get<1>(torch::linalg::eig(imag));
    imag = torch::transpose(imag, 2, 3).view({dim_i, -1, 3, 3}).contiguous();

    std::vector<signed long> dims = {dim_i, dim_eig*2, 3, 3}; 
    swp = torch::zeros(dims, op);  
    O   = torch::zeros(dims, op);
    L   = torch::zeros(dims, op);

    const dim3 blk__ = BLOCKS(threads, dim_i, dims[1]*3, 3); 
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(imag.scalar_type(), "intersections", ([&]
    {
        _intersectionK<scalar_t><<< blk__, threads >>>(
            O.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
            L.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
          swp.packed_accessor64<double, 4, torch::RestrictPtrTraits>(),

            A.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 
            G.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
         imag.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            dim_i, dims[1]); 
    })); 

    torch::Tensor diag = torch::pow(O.sum({-1}), 2) + torch::pow(L.sum({-1}), 2); 
    torch::Tensor id   = std::get<1>(diag.sort(-1, false)); 
    
    torch::Tensor diag_sol = torch::zeros({dim_i, dim_eig*2, 3   }, op); 
    torch::Tensor sols_vec = torch::zeros({dim_i, dim_eig*2, 3, 3}, op); 
    const dim3 blk_r = BLOCKS(threads, dim_i, dim_eig*2, 9);  
    AT_DISPATCH_FLOATING_TYPES(diag_sol.scalar_type(), "sols", ([&]
    {
        _SolsK<scalar_t><<< blk_r, threads >>>(
            diag_sol.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 
            sols_vec.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
            
                  id.packed_accessor32<long, 3, torch::RestrictPtrTraits>(),  
                diag.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 
                 swp.packed_accessor64<double, 4, torch::RestrictPtrTraits>(), 
                dim_i, dim_eig, null);  
    })); 

    cudaFree(sy); 
    cudaFree(sz); 
    cudaFree(dy); 
    cudaFree(dz); 

    return {sols_vec, diag_sol};
}

std::map<std::string, torch::Tensor> _Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, 
        torch::Tensor met_xy, torch::Tensor masses, 
        torch::Tensor sigma)
{
    torch::Tensor H = _Base_Matrix_H(pmc_b, pmc_mu, masses); 
    torch::Tensor shape = _Shape_Matrix(H, {0, 0, 1});
    sigma = _Expand_Matrix(H, sigma.view({-1, 2, 2})) + shape; 
    sigma = Operators::CUDA::Inverse(sigma) - shape;
    torch::Tensor X = std::get<0>(_DotMatrix(met_xy, H, sigma)); 
 
    const unsigned int dim_i = sigma.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3);
    AT_DISPATCH_FLOATING_TYPES(sigma.scalar_type(), "derivative", ([&]
    {
        _DerivativeK<scalar_t><<< blk, threads >>>(
            sigma.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            dim_i); 
    })); 

    sigma = Operators::CUDA::Mul(X, sigma); 
    AT_DISPATCH_FLOATING_TYPES(sigma.scalar_type(), "derivative", ([&]
    {
        _transSumK<scalar_t><<< blk, threads >>>(
            shape.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            sigma.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            dim_i); 
    })); 

    std::map<std::string, torch::Tensor> output;
    output["M"] = shape; 
    output["H"] = H;
    output["X"] = X; 
    return output; 
}

std::map<std::string, torch::Tensor> _Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, 
        torch::Tensor met_xy, torch::Tensor masses, 
        torch::Tensor sigma, const double null)
{
    std::map<std::string, torch::Tensor> nu; 
    nu = _Nu(pmc_b, pmc_mu, met_xy, masses, sigma); 

    torch::Tensor M = nu["M"]; 
    torch::Tensor H = nu["H"]; 
    torch::Tensor X = nu["X"]; 

    std::tuple<torch::Tensor, torch::Tensor> sols; 
    sols = _Intersection(M, _Shape_Matrix(M, {1, 1, -1}), null); 

    torch::Tensor sec = std::get<0>(sols);  
    const torch::TensorOptions op = _MakeOp(sec); 
    const unsigned int threads = 1024; 
    const unsigned int dim_i = sec.size(0);
    const unsigned int dim_eig = sec.size(1);  
    const unsigned int dim_j = 3; 

    std::vector<signed long> dims = {dim_i, dim_eig*3, dim_j}; 
    torch::Tensor sol_chi2 = torch::zeros(dims, op); 
    torch::Tensor sol_vecs = torch::zeros(dims, op); 

    const dim3 blk = BLOCKS(threads, dim_i, dim_eig, dim_j*dim_j); 
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "chi2", ([&]
    {
        _Y_dot_X_dot_Y<scalar_t><<< blk, threads >>>(
            sol_chi2.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 
            sol_vecs.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 

                   X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                 sec.packed_accessor64<double  , 4, torch::RestrictPtrTraits>(), 
            dim_i, dim_eig, dim_j); 
    })); 

    sol_chi2 = sol_chi2.sum(-1);
    torch::Tensor id = std::get<1>(sol_chi2.sort(-1, false)); 
    
    torch::Tensor _nu_v = torch::zeros(dims, op); 
    torch::Tensor _chi2 = torch::zeros_like(sol_chi2); 

    AT_DISPATCH_FLOATING_TYPES(H.scalar_type(), "Nu", ([&]
    {
        _NuK<scalar_t><<< blk, threads >>>(
          _nu_v.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 
          _chi2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 

             id.packed_accessor32<    long, 2, torch::RestrictPtrTraits>(), 
              H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

         sol_vecs.packed_accessor64<double, 3, torch::RestrictPtrTraits>(), 
         sol_chi2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
            dim_i, dim_eig, dim_j*dim_j); 
    })); 

    std::map<std::string, torch::Tensor> output; 
    int max_len = torch::max((_chi2 >= 0).sum(-1)).item<int>(); 
    output["NuVec"] = _nu_v.index({
            torch::indexing::Slice(), 
            torch::indexing::Slice(dims[1] - max_len, torch::indexing::None),
            torch::indexing::Slice()
    }); 
 
    output["chi2"]  = _chi2.index({
            torch::indexing::Slice(), 
            torch::indexing::Slice(dims[1] - max_len, torch::indexing::None)
    }); 
    return output;  
}

std::map<std::string, torch::Tensor> _NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_l1, torch::Tensor pmc_l2,
                torch::Tensor met_xy, torch::Tensor masses, 
                const double null)
{
    std::tuple<torch::Tensor, torch::Tensor> X;
    const torch::TensorOptions op = _MakeOp(masses); 
    const unsigned int threads = 1024;
    const unsigned int dim_j = 3;  
    const unsigned int dim_i = pmc_b1.size(0); 
    const dim3 blk   = BLOCKS(threads, dim_i, dim_j, 2); 
    const dim3 blk_  = BLOCKS(threads, dim_i,    27, 2);   
    const dim3 blk_d = BLOCKS(threads, dim_i,     9, 2); 
 
    torch::Tensor H1 = _Base_Matrix_H(pmc_b1, pmc_l1, masses); 
    torch::Tensor H2 = _Base_Matrix_H(pmc_b2, pmc_l2, masses); 
    torch::Tensor circl = _Shape_Matrix(H1, {1, 1, -1}); 
    torch::Tensor H_perp_1 = H1.clone(); 
    torch::Tensor H_perp_2 = H2.clone(); 

    torch::Tensor N1 = torch::zeros({dim_i, dim_j, dim_j, dim_j}, op); 
    torch::Tensor N2 = torch::zeros({dim_i, dim_j, dim_j, dim_j}, op); 

    AT_DISPATCH_FLOATING_TYPES(H1.scalar_type(), "H_perp", ([&]
    {
        _H_perp_K<scalar_t><<< blk, threads >>>(
            H_perp_1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            H_perp_2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            dim_i, dim_j, 2); 
    })); 

    torch::Tensor H_inv_1 = Operators::CUDA::Inverse(H_perp_1); 
    torch::Tensor H_inv_2 = Operators::CUDA::Inverse(H_perp_2);

    AT_DISPATCH_FLOATING_TYPES(H1.scalar_type(), "YT_DOT_X_DOTY", ([&]
    {
        _YT_dot_X_dot_Y<scalar_t><<< blk_, threads >>>(
            N1.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(), 
            N2.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(), 

       H_inv_1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
         circl.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

       H_inv_2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
         circl.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            dim_i, dim_j, 2); 
    })); 
    
    N1 = N1.sum(-1); 
    N2 = N2.sum(-1); 
 
    X = _DotMatrix(met_xy, circl, N2); 
    torch::Tensor n_ = std::get<0>(X); 
    torch::Tensor S  = std::get<1>(X); 
    
    X = _Intersection(N1, n_, null);
    torch::Tensor sol  = std::get<0>(X); 
    torch::Tensor diag = std::get<1>(X).view({dim_i, -1});
    torch::Tensor id   = std::get<1>(diag.sort(-1, false)); 
    
    const unsigned int dim_eig = std::get<0>(X).size(1);
    std::vector<signed long> dims = {dim_i, dim_eig*3, dim_j}; 
    torch::Tensor v    = torch::zeros(dims, op);
    torch::Tensor v_   = torch::zeros(dims, op);

    torch::Tensor nu   = torch::zeros(dims, op); 
    torch::Tensor nu_  = torch::zeros(dims, op); 
    torch::Tensor dnu  = torch::zeros({dim_i, dim_eig*3}, op); 

    torch::Tensor K1   = torch::zeros({dim_i, dim_j, dim_j}, op);
    torch::Tensor K2   = torch::zeros({dim_i, dim_j, dim_j}, op);
    
    const dim3 blk__ = BLOCKS(threads, dim_i, dims[1], 6);   
    AT_DISPATCH_FLOATING_TYPES(N1.scalar_type(), "DOTS", ([&]
    {
        _DotK<scalar_t><<< blk__, threads >>>(
             v.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            v_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

             S.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),  
           sol.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(), 
            dim_i, dims[1], 6); 

        _K_Kern<scalar_t><<< blk_d, threads >>>(
            K1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            K2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

            H1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
       H_inv_1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

            H2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
       H_inv_2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            dim_i, 9, 2); 

        _NuNuK<scalar_t><<< blk__, threads >>>(
            nu.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
           nu_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
           dnu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 

            K1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            K2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

             v.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            v_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

            id.packed_accessor32<long    , 2, torch::RestrictPtrTraits>(), 
          diag.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            dim_i, dims[1], 6); 
    })); 
   
    std::map<std::string, torch::Tensor> output; 
    torch::Tensor none = (nu.sum(-1) != 0).sum(-1);  
    int max_len = torch::max(none).item<int>(); 
    output["NuVec_1"] = nu.index({
            torch::indexing::Slice(), 
            torch::indexing::Slice(0, max_len), 
            torch::indexing::Slice()
    }); 
    output["NuVec_2"] = nu_.index({
            torch::indexing::Slice(), 
            torch::indexing::Slice(0, max_len), 
            torch::indexing::Slice()
    }); 
    output["diagonal"] = dnu.index({
            torch::indexing::Slice(), 
            torch::indexing::Slice(0, max_len)
    });   
    
    output["n_"] = n_; 
    output["H_perp_1"] = H_perp_1; 
    output["H_perp_2"] = H_perp_2;
    output["NoSols"] = none == 0;   
    return output; 
}

#endif
