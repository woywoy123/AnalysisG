#include <transform/cartesian-cuda/cartesian.h>
#include <transform/polar-cuda/polar.h>
#include <torch/torch.h>
#include <operators.h>
#include <physics.h>
#include <stdio.h>
#include <vector>
#include <map>

#ifndef NUSOL_CUDA_KERNEL_H
#define NUSOL_CUDA_KERNEL_H
#include "kernel.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static const dim3 BLOCKS(
    const unsigned int threads, const unsigned int len, 
    const unsigned int dy, const unsigned int dz)
{
    const dim3 blocks( (len + threads -1) / threads, dy, dz); 
    return blocks; 
}

static const dim3 BLOCKS(const unsigned int threads, const unsigned int dx)
{
    const dim3 blocks( (dx + threads -1) / threads); 
    return blocks; 
}

static const dim3 BLOCKS(const unsigned int threads, 
        const unsigned int dx, const unsigned int dy)
{
    const dim3 blocks( (dx + threads -1) / threads, dy); 
    return blocks; 
}

static const torch::TensorOptions _MakeOp(torch::Tensor v1)
{
    return torch::TensorOptions().dtype(v1.scalar_type()).device(v1.device()); 
}

static const std::map<std::string, torch::Tensor> _convert(torch::Tensor met_phi)
{   
    torch::Tensor met = met_phi.index({torch::indexing::Slice(), 0}); 
    torch::Tensor phi = met_phi.index({torch::indexing::Slice(), 1}); 
    torch::Tensor met_x = Transform::CUDA::Px(met, phi);
    torch::Tensor met_y = Transform::CUDA::Py(met, phi);

    std::map<std::string, torch::Tensor> out;
    out["met_xy"] = torch::cat({met_x, met_y}, -1); 
    return out; 
}

static const std::map<std::string, torch::Tensor> _convert(torch::Tensor pmu1, torch::Tensor pmu2)
{  
    const unsigned int dim_i = pmu1.size(0); 
    torch::Tensor com = torch::cat({pmu1, pmu2}, 0); 
    com = Transform::CUDA::PxPyPzE(com);

    std::map<std::string, torch::Tensor> out;
    out["pmc1"] = com.index({torch::indexing::Slice(0, dim_i)}); 
    out["pmc2"] = com.index({torch::indexing::Slice(dim_i, dim_i*2)}); 
    return out; 
}

static const torch::Tensor _format(std::vector<torch::Tensor> v)
{  
    std::vector<torch::Tensor> out; 
    for (torch::Tensor i : v){out.push_back(i.view({-1, 1}));}
    return torch::cat(out, -1); 
}

static const torch::Tensor _Shape_Matrix(torch::Tensor inpt, std::vector<long> vec)
{
    const unsigned int len_i = inpt.size(0); 
    const unsigned int len_j = vec.size(); 
    const unsigned int threads = 1024; 
    const torch::TensorOptions op = _MakeOp(inpt); 

    torch::Tensor out = torch::zeros_like(inpt); 
    torch::Tensor vecT = torch::zeros({1, 1, len_j}, op).to(torch::kCPU); 
    for (unsigned int i(0); i < len_j; ++i){ vecT[0][0][i] += vec[i]; }
    vecT = vecT.to(op); 

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

static unsigned int _getDim(torch::Tensor inpt, int index1, int index2, double step_size)
{
    torch::Tensor dx;
    dx = inpt.index({torch::indexing::Slice(), index1});
    dx -= inpt.index({torch::indexing::Slice(), index2});
    dx = torch::abs(dx)/step_size; 
    return torch::max(dx).item<int>(); 
}



static const torch::Tensor _Expand_Matrix(torch::Tensor inpt, torch::Tensor source)
{
    const unsigned int threads = 1024; 
    const unsigned int len_i = inpt.size(0);
    const unsigned int len_k = source.size(1); 
    const torch::TensorOptions op = _MakeOp(inpt); 
    source = source.view({source.size(0), len_k, -1}); 
  
    const unsigned int len_j = source.size(2); 
    const dim3 blk = BLOCKS(threads, len_i, len_k, len_j);
    torch::Tensor out = torch::zeros_like(inpt, op); 

    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]
    {
        _ShapeKernel<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                source.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, (len_k > inpt.size(1)) ? inpt.size(1) : len_k, len_j, true); 
    })); 

    return out; 
}


static const std::tuple<torch::Tensor, torch::Tensor> _DotMatrix(
        torch::Tensor MET_xy, torch::Tensor H, torch::Tensor Shape)
{
    const unsigned int threads = 1024; 
    const unsigned int dim_i = MET_xy.size(0); 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3);
    const torch::TensorOptions op = _MakeOp(H); 

    MET_xy = _Expand_Matrix(H, MET_xy); 
    torch::Tensor   X = torch::zeros({dim_i, 3, 3}, op);
    torch::Tensor dNu = torch::zeros({dim_i, 3, 3}, op); 

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

torch::Tensor _Base_Matrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu)
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

        _Base_Matrix_Nan<scalar_t><<< blk, threads>>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                len_i, len_m); 
    })); 
    return out; 
} 

torch::Tensor _Base_Matrix_H(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses)
{
    torch::Tensor base  = _Base_Matrix(pmc_b, pmc_mu, masses);
    const torch::TensorOptions op = _MakeOp(base); 
    const unsigned int threads = 1024; 
    const unsigned int dim_i = pmc_b.size(0); 
    const dim3 blk  = BLOCKS(threads, dim_i, 6, 3); 
    const dim3 blk_ = BLOCKS(threads, dim_i, 3, 3); 
    const dim3 blk_dot  = BLOCKS(threads, dim_i, 9, 3);   

    torch::Tensor phi   = Transform::CUDA::Phi(pmc_mu); 
    torch::Tensor theta = Physics::CUDA::Theta(pmc_mu); 

    torch::Tensor Rx = _Expand_Matrix(base, pmc_b); 
    torch::Tensor Rx_ = torch::zeros({dim_i, 3, 3, 3}, op); 
    torch::Tensor Rz  = torch::zeros({dim_i, 3, 3}, op); 
    torch::Tensor Ry  = torch::zeros({dim_i, 3, 3}, op); 

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

        _Rz_Rx_Ry_dot_K<scalar_t><<< blk_dot, threads >>>(
               Rx_.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(), 
                Rx.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                Ry.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                Rz.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                dim_i, 3, 3); 
    })); 
    Rx = Rx_.sum(-1); 

    AT_DISPATCH_FLOATING_TYPES(phi.scalar_type(), "BaseMatrixH", ([&]
    {
        _Base_Matrix_H_Kernel<scalar_t><<< blk_, threads >>>(
               RxT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                Rx.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
               dim_i);

        _Rz_Rx_Ry_dot_K<scalar_t><<< blk_dot, threads >>>(
                Rx_.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(), 
                RxT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                RzT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                RyT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                dim_i, 3, 3); 
        
        _dot_K<scalar_t><<< blk_, threads >>>(
                Rx.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                Rx_.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(), 
                base.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                dim_i, 3, 3); 
    })); 
    return Rx; 
}

std::tuple<torch::Tensor, torch::Tensor> _Intersection(torch::Tensor A, torch::Tensor B, const double null)
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

    std::tuple<torch::Tensor, torch::Tensor> v = Operators::CUDA::Inverse(A, true);
    torch::Tensor ignore = std::get<1>(v); 
    torch::Tensor imag = std::get<0>(v); 

    imag = Operators::CUDA::Mul(imag, B);
    imag = torch::linalg::eigvals(imag); 
    
    const torch::TensorOptions op = _MakeOp(A); 
    const unsigned int dim_eig = imag.size(-1); 
    const dim3 blk_ = BLOCKS(threads, dim_i, 9, dim_eig); 
    std::vector<signed long> dims = {dim_i, dim_eig, 3, 3}; 

    torch::Tensor G   = torch::zeros(dims, op); 
    torch::Tensor L   = torch::zeros(dims, op);
    torch::Tensor O   = torch::zeros(dims, op);
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

    uint8_t dev = G.get_device(); 
    cudaSetDevice(dev); 
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

    dims = {dim_i, dim_eig*2, 3, 3}; 
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
              ignore.packed_accessor64<bool, 2, torch::RestrictPtrTraits>(), 
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
    torch::Tensor _chi2 = torch::zeros_like(sol_chi2, op); 

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
    std::map<std::string, torch::Tensor> output; 
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
    const unsigned int dim_eig = std::get<0>(X).size(1);
    std::vector<signed long> dims = {dim_i, dim_eig*3, dim_j}; 

    torch::Tensor sol  = std::get<0>(X); 
    torch::Tensor diag = std::get<1>(X).view({dim_i, -1});
    torch::Tensor id   = std::get<1>(diag.sort(-1, false)); 
    
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

std::map<std::string, torch::Tensor> _NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2,
                torch::Tensor mass_t1, torch::Tensor mass_t2, torch::Tensor mass_w1, torch::Tensor mass_w2,
                torch::Tensor met_xy, 
                const double null)
{
    std::tuple<torch::Tensor, torch::Tensor> X;
    std::map<std::string, torch::Tensor> output; 
    const torch::TensorOptions op = _MakeOp(mass_w1); 
    const unsigned int threads = 1024;
    const unsigned int dim_j = 3;  
    const unsigned int dim_i = pmc_b1.size(0); 
    const dim3 blk   = BLOCKS(threads, dim_i, dim_j, 2); 
    const dim3 blk_  = BLOCKS(threads, dim_i,    27, 2);   
    const dim3 blk_d = BLOCKS(threads, dim_i,     9, 2); 

    torch::Tensor mass_1 = torch::cat({mass_w1, mass_t1, torch::zeros_like(mass_w1)}, -1).to(op.requires_grad(false));
    torch::Tensor mass_2 = torch::cat({mass_w2, mass_t2, torch::zeros_like(mass_w2)}, -1).to(op.requires_grad(false));

    torch::Tensor H1 = _Base_Matrix_H(pmc_b1, pmc_l1, mass_1); 
    torch::Tensor H2 = _Base_Matrix_H(pmc_b2, pmc_l2, mass_2); 

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
    const unsigned int dim_eig = std::get<0>(X).size(1);
    std::vector<signed long> dims = {dim_i, dim_eig*3, dim_j}; 

    torch::Tensor sol  = std::get<0>(X); 
    torch::Tensor diag = std::get<1>(X).view({dim_i, -1});
    torch::Tensor id   = std::get<1>(diag.sort(-1, false)); 
    
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

std::map<std::string, torch::Tensor> _NuNuCombinatorial(
        torch::Tensor edge_index, torch::Tensor batch, torch::Tensor pmc, torch::Tensor pid, 
        torch::Tensor met_xy, torch::Tensor met_updown, torch::Tensor mass_nom, torch::Tensor mass_updown, 
        const double step_size, const double null)       
{
    const torch::TensorOptions op = _MakeOp(edge_index).dtype(torch::kDouble); 
    const unsigned int threads = 1024; 
    const unsigned int dim_i = edge_index.size(1);
    const dim3 blk = BLOCKS(threads, dim_i, dim_i); 
    const dim3 blk_ = BLOCKS(threads, dim_i, 4, 4); 

    torch::Tensor pid_ = pid.to(torch::kDouble).contiguous(); 
    torch::Tensor batch_ = batch.to(torch::kDouble).contiguous();

    torch::Tensor pmc_b1 = torch::zeros({dim_i, 4}, op); 
    torch::Tensor pmc_l1 = torch::zeros({dim_i, 4}, op); 

    torch::Tensor pmc_b2 = torch::zeros({dim_i, 4}, op); 
    torch::Tensor pmc_l2 = torch::zeros({dim_i, 4}, op); 

    torch::Tensor edge_idx = (-1)*torch::ones({dim_i}, op); 
    torch::Tensor batch_idx = (-1)*torch::ones({dim_i}, op);    
    torch::Tensor l1l2_b1b2 = (-1)*torch::ones({dim_i, 4}, op);
    torch::Tensor edge_ = torch::transpose(edge_index, 0, 1).to(torch::kDouble).contiguous();
    
    AT_DISPATCH_FLOATING_TYPES(edge_.scalar_type(), "msknunu", ([&]
    {
        _NuNuMask<scalar_t><<< blk, threads >>>(
            l1l2_b1b2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),                 
             edge_idx.packed_accessor64<double, 1, torch::RestrictPtrTraits>(), 
                edge_.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                 pid_.packed_accessor64<double, 2, torch::RestrictPtrTraits>(), 
            dim_i); 

        _NuNuPopulate<scalar_t><<< blk_, threads >>>(
                pmc_b1.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                pmc_b2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                pmc_l1.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                pmc_l2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
             batch_idx.packed_accessor64<double, 1, torch::RestrictPtrTraits>(),

                   pmc.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),
                batch_.packed_accessor64<double, 1, torch::RestrictPtrTraits>(),
             l1l2_b1b2.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),                 
              edge_idx.packed_accessor64<double, 1, torch::RestrictPtrTraits>(), 
                dim_i, 4, 4);       
    })); 


    torch::Tensor msk = edge_idx > -1;
    torch::Tensor edge_idx_ = edge_idx.index({msk}).contiguous(); 
    torch::Tensor batch_idx_ = batch_idx.index({msk}).contiguous().to(torch::kLong);

    pmc_b1 = pmc_b1.index({msk}).clone(); 
    pmc_l1 = pmc_l1.index({msk}).clone(); 

    pmc_b2 = pmc_b2.index({msk}).clone(); 
    pmc_l2 = pmc_l2.index({msk}).clone(); 

    torch::Tensor _nu1 = torch::zeros_like(pmc_l1).index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3)}); 
    torch::Tensor _nu2 = torch::zeros_like(pmc_l1).index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3)}); 

    std::map<std::string, torch::Tensor> output;
    torch::Tensor mass_t1 = torch::ones({edge_idx_.size(0), 1}, torch::TensorOptions().dtype(pmc_l1.dtype()).device(pmc_l1.device()).requires_grad(true)); 
    torch::Tensor mass_w1 = torch::ones({edge_idx_.size(0), 1}, torch::TensorOptions().dtype(pmc_l1.dtype()).device(pmc_l1.device()).requires_grad(true)); 

    torch::Tensor mass_t2 = torch::ones({edge_idx_.size(0), 1}, torch::TensorOptions().dtype(pmc_l1.dtype()).device(pmc_l1.device()).requires_grad(true)); 
    torch::Tensor mass_w2 = torch::ones({edge_idx_.size(0), 1}, torch::TensorOptions().dtype(pmc_l1.dtype()).device(pmc_l1.device()).requires_grad(true)); 

    torch::Tensor met_xy_m = met_xy.index({batch_idx_}).view({-1, 1, 2}); 
    torch::Tensor met_null = torch::zeros_like(met_xy_m); 
    torch::Tensor mass_ = torch::zeros_like(pmc_l1); 

    torch::optim::AdamOptions set(0.5); 
    torch::optim::Adam opti({mass_t1, mass_t2, mass_w1, mass_w2}, set); 

    for (int i(0); i < 10; ++i){
        std::map<std::string, torch::Tensor> res = _NuNu(pmc_b1, pmc_b2, pmc_l1, pmc_l2, mass_t1*172.62, mass_t2*172.62, mass_w1*80.385, mass_w2*80.385, met_xy_m, null); 
        torch::Tensor is_sol = res["NoSols"] != false; 
        torch::Tensor nu1_v = res["NuVec_1"].index({torch::indexing::Slice(), 0});
        torch::Tensor nu2_v = res["NuVec_2"].index({torch::indexing::Slice(), 0});

        torch::Tensor met_;
        met_  = nu1_v.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 2)}).view({-1, 1, 2});
        met_ += nu2_v.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 2)}).view({-1, 1, 2});
        met_ -= met_xy_m; 
       
        std::cout << met_ << std::endl;
//        opti.zero_grad(); 
        torch::Tensor loss = torch::nn::functional::mse_loss(met_, met_null);
        std::cout << loss << std::endl; 
//        loss.backward();
//        opti.step(); 

        _nu1.index_put_({is_sol}, nu1_v.index({is_sol})); 
        _nu2.index_put_({is_sol}, nu2_v.index({is_sol})); 

        mass_.index_put_({is_sol}, torch::cat({mass_t1*172.62, mass_t2*172.62, mass_w1*80.385, mass_w2*80.385}, -1).index({is_sol})); 
    } 
    output["nu1"] = _nu1; 
    output["nu2"] = _nu2;  
    output["masses"] = mass_; 
    return output; 
}




// -------------------- Interfaces ------------------------- //
std::map<std::string, torch::Tensor> _NuPolar(
        torch::Tensor pmu_b, torch::Tensor pmu_mu, 
        torch::Tensor met_phi, torch::Tensor masses, 
        torch::Tensor sigma, const double null)
{
    std::map<std::string, torch::Tensor> met = _convert(met_phi); 
    std::map<std::string, torch::Tensor> pmc = _convert(pmu_b, pmu_mu); 

    return _Nu(pmc["pmc1"], pmc["pmc2"], met["met_xy"], masses, sigma, null);
}

std::map<std::string, torch::Tensor> _NuPolar(
        torch::Tensor pt_b, torch::Tensor eta_b, torch::Tensor phi_b, torch::Tensor e_b, 
        torch::Tensor pt_mu, torch::Tensor eta_mu, torch::Tensor phi_mu, torch::Tensor e_mu, 
        torch::Tensor met, torch::Tensor phi, torch::Tensor masses, 
        torch::Tensor sigma, const double null)
{
    torch::Tensor pmu_b = _format({pt_b, eta_b, phi_b, e_b}); 
    torch::Tensor pmu_mu = _format({pt_mu, eta_mu, phi_mu, e_mu}); 
    torch::Tensor met_ = _format({met, phi}); 
    
    std::map<std::string, torch::Tensor> pmc = _convert(pmu_b, pmu_mu); 
    std::map<std::string, torch::Tensor> _met = _convert(met_);
    return _Nu(pmc["pmc1"], pmc["pmc2"], _met["met_xy"], masses, sigma, null);
}


std::map<std::string, torch::Tensor> _NuCart(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, 
        torch::Tensor met_xy, torch::Tensor masses, 
        torch::Tensor sigma, const double null)
{
    return _Nu(pmc_b, pmc_mu, met_xy, masses, sigma, null);
}

std::map<std::string, torch::Tensor> _NuCart(
        torch::Tensor px_b, torch::Tensor py_b, torch::Tensor pz_b, torch::Tensor e_b, 
        torch::Tensor px_mu, torch::Tensor py_mu, torch::Tensor pz_mu, torch::Tensor e_mu, 
        torch::Tensor metx, torch::Tensor mety, torch::Tensor masses, 
        torch::Tensor sigma, const double null)
{
    torch::Tensor pmc_b = _format({px_b, py_b, pz_b, e_b}); 
    torch::Tensor pmc_mu = _format({px_mu, py_mu, pz_mu, e_mu}); 
    torch::Tensor met_ = _format({metx, mety}); 
    
    return _Nu(pmc_b, pmc_mu, met_, masses, sigma, null);
}

std::map<std::string, torch::Tensor> _NuNuPolar(
        torch::Tensor pmu_b1 , torch::Tensor pmu_b2, 
        torch::Tensor pmu_mu1, torch::Tensor pmu_mu2, 
        torch::Tensor met_phi, torch::Tensor masses, 
        const double null)
{
    std::map<std::string, torch::Tensor> met = _convert(met_phi); 
    std::map<std::string, torch::Tensor> pmc_b  = _convert(pmu_b1, pmu_b2); 
    std::map<std::string, torch::Tensor> pmc_mu = _convert(pmu_mu1, pmu_mu2); 

    return _NuNu(pmc_b["pmc1"], pmc_b["pmc2"], pmc_mu["pmc1"], pmc_mu["pmc2"], met["met_xy"], masses, null);
}

std::map<std::string, torch::Tensor> _NuNuPolar(
        torch::Tensor pt_b1, torch::Tensor eta_b1, torch::Tensor phi_b1, torch::Tensor e_b1, 
        torch::Tensor pt_b2, torch::Tensor eta_b2, torch::Tensor phi_b2, torch::Tensor e_b2, 

        torch::Tensor pt_mu1, torch::Tensor eta_mu1, torch::Tensor phi_mu1, torch::Tensor e_mu1, 
        torch::Tensor pt_mu2, torch::Tensor eta_mu2, torch::Tensor phi_mu2, torch::Tensor e_mu2, 

        torch::Tensor met, torch::Tensor phi, 
        torch::Tensor masses, const double null)
{
    torch::Tensor pmu_b1 = _format({pt_b1, eta_b1, phi_b1, e_b1});
    torch::Tensor pmu_b2 = _format({pt_b2, eta_b2, phi_b2, e_b2});

    torch::Tensor pmu_mu1 = _format({pt_mu1, eta_mu1, phi_mu1, e_mu1});
    torch::Tensor pmu_mu2 = _format({pt_mu2, eta_mu2, phi_mu2, e_mu2});

    std::map<std::string, torch::Tensor> _met   = _convert(_format({met, phi}));  
    std::map<std::string, torch::Tensor> pmc_b  = _convert(pmu_b1 , pmu_b2); 
    std::map<std::string, torch::Tensor> pmc_mu = _convert(pmu_mu1, pmu_mu2); 

    return _NuNu(pmc_b["pmc1"], pmc_b["pmc2"], pmc_mu["pmc1"], pmc_mu["pmc2"], _met["met_xy"], masses, null);
}

std::map<std::string, torch::Tensor> _NuNuCart(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
        torch::Tensor pmc_mu1, torch::Tensor pmc_mu2,
        torch::Tensor met_xy, torch::Tensor masses, const double null)
{
    return _NuNu(pmc_b1, pmc_b2, pmc_mu1, pmc_mu2, met_xy, masses, null);
}

std::map<std::string, torch::Tensor> _NuNuCart(
        torch::Tensor px_b1, torch::Tensor py_b1, torch::Tensor pz_b1, torch::Tensor e_b1, 
        torch::Tensor px_b2, torch::Tensor py_b2, torch::Tensor pz_b2, torch::Tensor e_b2, 

        torch::Tensor px_mu1, torch::Tensor py_mu1, torch::Tensor pz_mu1, torch::Tensor e_mu1, 
        torch::Tensor px_mu2, torch::Tensor py_mu2, torch::Tensor pz_mu2, torch::Tensor e_mu2, 

        torch::Tensor metx, torch::Tensor mety, torch::Tensor masses, const double null)
{
    torch::Tensor pmc_b1  = _format({px_b1, py_b1, pz_b1, e_b1}); 
    torch::Tensor pmc_b2  = _format({px_b2, py_b2, pz_b2, e_b2}); 
    torch::Tensor pmc_mu1 = _format({px_mu1, py_mu1, pz_mu1, e_mu1}); 
    torch::Tensor pmc_mu2 = _format({px_mu2, py_mu2, pz_mu2, e_mu2}); 
    torch::Tensor met_    = _format({metx, mety}); 
    
    return _NuNu(pmc_b1, pmc_b2, pmc_mu1, pmc_mu2, met_, masses, null);
}


std::map<std::string, torch::Tensor> _NuNuCombinatorialPolar(
        torch::Tensor edge_index, torch::Tensor batch, torch::Tensor pmu, torch::Tensor pid, 
        torch::Tensor met, torch::Tensor met_updown, torch::Tensor masses, torch::Tensor masses_updown, 
        const double step_size, const double null)
{
    torch::Tensor pmc = Transform::CUDA::PxPyPzE(pmu); 
    torch::Tensor met_xy = std::map<std::string, torch::Tensor>(_convert(met))["met_xy"]; 
    return _NuNuCombinatorial(edge_index, batch, pmc, pid, met_xy, met_updown, masses, masses_updown, step_size, null); 
}

#endif
