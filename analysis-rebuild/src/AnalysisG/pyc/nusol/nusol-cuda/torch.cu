#include <operators/operators-cuda.h>
#include <transform/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <physics/physics-cuda.h>
#include <torch/torch.h>
#include <stdio.h>
#include <vector>
#include <map>

#ifndef NUSOL_CUDA_KERNEL_H
#define NUSOL_CUDA_KERNEL_H
#include "kernel.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const dim3 BLOCKS(
    const unsigned int threads, const unsigned int len, 
    const unsigned int dy, const unsigned int dz)
{
    const dim3 blocks( (len + threads -1) / threads, dy, dz); 
    return blocks; 
}

const dim3 BLOCKS(const unsigned int threads, const unsigned int dx){
    const dim3 blocks( (dx + threads -1) / threads); 
    return blocks; 
}

const dim3 BLOCKS(const unsigned int threads, const unsigned int dx, const unsigned int dy){
    const dim3 blocks( (dx + threads -1) / threads, dy); 
    return blocks; 
}

const torch::TensorOptions _MakeOp(torch::Tensor v1){
    return torch::TensorOptions().dtype(v1.scalar_type()).device(v1.device()); 
}

const std::map<std::string, torch::Tensor> _convert(torch::Tensor met_phi){   
    torch::Tensor met = met_phi.index({torch::indexing::Slice(), 0}); 
    torch::Tensor phi = met_phi.index({torch::indexing::Slice(), 1}); 
    torch::Tensor met_x = transform::cuda::Px(met, phi);
    torch::Tensor met_y = transform::cuda::Py(met, phi);

    std::map<std::string, torch::Tensor> out;
    out["met_xy"] = torch::cat({met_x, met_y}, -1); 
    return out; 
}

std::map<std::string, torch::Tensor> _convert(torch::Tensor pmu1, torch::Tensor pmu2){  
    const unsigned int dim_i = pmu1.size(0); 
    torch::Tensor com = torch::cat({pmu1, pmu2}, 0); 
    com = transform::cuda::PxPyPzE(com);

    std::map<std::string, torch::Tensor> out;
    out["pmc1"] = com.index({torch::indexing::Slice(0, dim_i)}); 
    out["pmc2"] = com.index({torch::indexing::Slice(dim_i, dim_i*2)}); 
    return out; 
}

torch::Tensor _format(std::vector<torch::Tensor> v){  
    std::vector<torch::Tensor> out; 
    for (torch::Tensor i : v){out.push_back(i.view({-1, 1}));}
    return torch::cat(out, -1); 
}

torch::Tensor _Shape_Matrix(torch::Tensor inpt, std::vector<long> vec){
    const unsigned int len_i = inpt.size(0); 
    const unsigned int len_j = vec.size(); 
    const unsigned int threads = 1024; 
    const torch::TensorOptions op = _MakeOp(inpt); 

    torch::Tensor out = torch::zeros_like(inpt); 
    torch::Tensor vecT = torch::zeros({1, 1, len_j}, op).to(torch::kCPU); 
    for (unsigned int i(0); i < len_j; ++i){ vecT[0][0][i] += vec[i]; }
    vecT = vecT.to(op); 

    const dim3 blk = BLOCKS(threads, len_i, len_j, len_j);
    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]{
        _ShapeKernel<scalar_t><<< blk, threads >>>(
                 out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                vecT.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, len_j, len_j, false); 
    })); 

    return out; 
} 

unsigned int _getDim(torch::Tensor inpt, int index1, int index2, double step_size){
    torch::Tensor dx;
    dx = inpt.index({torch::indexing::Slice(), index1});
    dx -= inpt.index({torch::indexing::Slice(), index2});
    dx = torch::abs(dx)/step_size; 
    return torch::max(dx).item<int>(); 
}

const torch::Tensor _Expand_Matrix(torch::Tensor inpt, torch::Tensor source){
    const unsigned int threads = 1024; 
    const unsigned int len_i = inpt.size(0);
    const unsigned int len_k = source.size(1); 
    const torch::TensorOptions op = _MakeOp(inpt); 
    source = source.view({source.size(0), len_k, -1}); 
  
    const unsigned int len_j = source.size(2); 
    const dim3 blk = BLOCKS(threads, len_i, len_k, len_j);
    torch::Tensor out = torch::zeros_like(inpt, op); 

    AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "ShapeMatrix", ([&]{
        _ShapeKernel<scalar_t><<< blk, threads >>>(
                out.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                source.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                len_i, (len_k > inpt.size(1)) ? inpt.size(1) : len_k, len_j, true); 
    })); 

    return out; 
}


std::tuple<torch::Tensor, torch::Tensor> _DotMatrix(torch::Tensor MET_xy, torch::Tensor H, torch::Tensor Shape){
    const unsigned int threads = 1024; 
    const unsigned int dim_i = MET_xy.size(0); 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3);
    const torch::TensorOptions op = _MakeOp(H); 

    MET_xy = _Expand_Matrix(H, MET_xy); 
    torch::Tensor   X = torch::zeros({dim_i, 3, 3}, op);
    torch::Tensor dNu = torch::zeros({dim_i, 3, 3}, op); 

    AT_DISPATCH_FLOATING_TYPES(MET_xy.scalar_type(), "NuMatrix", ([&]{
        _V0_deltaK<scalar_t><<< blk, threads >>>(
                 X.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
               dNu.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            MET_xy.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
             Shape.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                 H.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                 dim_i); 
    })); 

    return {operators::cuda::Mul(X, dNu), dNu}; 
}

torch::Tensor _Base_Matrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses_W_top_nu){
    const unsigned int threads = 1024; 
    const unsigned int len_i = pmc_b.size(0); 
    const unsigned int len_m = masses_W_top_nu.size(0); 

    torch::Tensor beta2_b   = physics::cuda::Beta2(pmc_b); 
    torch::Tensor mass2_b   = physics::cuda::M2(pmc_b); 
    
    torch::Tensor beta2_mu  = physics::cuda::Beta2(pmc_mu);
    torch::Tensor mass2_mu  = physics::cuda::M2(pmc_mu); 
    
    torch::Tensor costheta  = operators::cuda::CosTheta(pmc_b, pmc_mu, 3);

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

torch::Tensor _Base_Matrix_H(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses){
    torch::Tensor base  = _Base_Matrix(pmc_b, pmc_mu, masses);
    const torch::TensorOptions op = _MakeOp(base); 
    const unsigned int threads = 1024; 
    const unsigned int dim_i = pmc_b.size(0); 
    const dim3 blk      = BLOCKS(threads, dim_i, 6, 3); 
    const dim3 blk_     = BLOCKS(threads, dim_i, 3, 3); 
    const dim3 blk_dot  = BLOCKS(threads, dim_i, 9, 3);   

    torch::Tensor phi   = transform::cuda::Phi(pmc_mu); 
    torch::Tensor theta = physics::cuda::Theta(pmc_mu); 

    torch::Tensor Rx = _Expand_Matrix(base, pmc_b); 
    torch::Tensor Rx_ = torch::zeros({dim_i, 3, 3, 3}, op); 
    torch::Tensor Rz  = torch::zeros({dim_i, 3, 3}, op); 
    torch::Tensor Ry  = torch::zeros({dim_i, 3, 3}, op); 

    torch::Tensor RzT = torch::zeros({dim_i, 3, 3}, op); 
    torch::Tensor RyT = torch::zeros({dim_i, 3, 3}, op); 
    torch::Tensor RxT = torch::zeros({dim_i, 3, 3}, op);

    AT_DISPATCH_FLOATING_TYPES(phi.scalar_type(), "BaseMatrixH", ([&]{
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

    AT_DISPATCH_FLOATING_TYPES(phi.scalar_type(), "BaseMatrixH", ([&]{
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

std::tuple<torch::Tensor, torch::Tensor> _Intersection(torch::Tensor A, torch::Tensor B, const double null){
    const unsigned int dim_i = A.size(0); 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, dim_i, 3, 3); 
   
    torch::Tensor det_A = operators::cuda::Determinant(A); 
    torch::Tensor det_B = operators::cuda::Determinant(B); 
    
    AT_DISPATCH_FLOATING_TYPES(det_A.scalar_type(), "Swap", ([&]{
        _SwapAB_K<scalar_t><<< blk, threads >>>(
                det_A.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                det_B.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                    A.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                    B.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                dim_i); 
    })); 

    std::tuple<torch::Tensor, torch::Tensor> v = operators::cuda::Inverse(A, true);
    torch::Tensor ignore = std::get<1>(v); 
    torch::Tensor imag = std::get<0>(v); 

    imag = operators::cuda::Mul(imag, B);
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
    torch::Tensor id   = std::get<1>(torch::log(diag).sort(-1, false)); 
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
    sigma = operators::cuda::Inverse(sigma) - shape;
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

    sigma = operators::cuda::Mul(X, sigma); 
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
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2,
        torch::Tensor masses1, torch::Tensor masses2, torch::Tensor met_xy, const double null
){
    std::tuple<torch::Tensor, torch::Tensor> X;
    std::map<std::string, torch::Tensor> output; 
    const torch::TensorOptions op = _MakeOp(masses1); 
    const unsigned int threads = 1024;
    const unsigned int dim_j = 3;  
    const unsigned int dim_i = pmc_b1.size(0); 
    const dim3 blk   = BLOCKS(threads, dim_i, dim_j, 2); 
    const dim3 blk_  = BLOCKS(threads, dim_i,    27, 2);   
    const dim3 blk_d = BLOCKS(threads, dim_i,     9, 2); 

    torch::Tensor H1 = _Base_Matrix_H(pmc_b1, pmc_l1, masses1); 
    torch::Tensor H2 = _Base_Matrix_H(pmc_b2, pmc_l2, masses2); 

    torch::Tensor circl = _Shape_Matrix(H1, {1, 1, -1}); 
    torch::Tensor H_perp_1 = H1.clone(); 
    torch::Tensor H_perp_2 = H2.clone(); 

    torch::Tensor N1 = torch::zeros({dim_i, dim_j, dim_j, dim_j}, op); 
    torch::Tensor N2 = torch::zeros({dim_i, dim_j, dim_j, dim_j}, op); 

    AT_DISPATCH_FLOATING_TYPES(H1.scalar_type(), "H_perp", ([&]{
        _H_perp_K<scalar_t><<< blk, threads >>>(
            H_perp_1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            H_perp_2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
            dim_i, dim_j, 2); 
    })); 

    torch::Tensor H_inv_1 = operators::cuda::Inverse(H_perp_1); 
    torch::Tensor H_inv_2 = operators::cuda::Inverse(H_perp_2);

    AT_DISPATCH_FLOATING_TYPES(H1.scalar_type(), "YT_DOT_X_DOTY", ([&]{
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
    torch::Tensor id   = std::get<1>(torch::log(diag).sort(-1, false)); 
    
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
    if (!max_len){max_len = 1;}
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



void _MassMatrix(
        const double mtop_l, const double mtop_u, 
        const double mw_l, const double mw_u, 
        torch::Tensor masses, const float steps)
{
    const float step_t = (mtop_u - mtop_l)/steps; 
    const float step_w = (mw_u - mw_l)/steps; 
    const unsigned int threads = 1024; 
    const dim3 blk = BLOCKS(threads, (int)steps, 2); 

    AT_DISPATCH_ALL_TYPES(masses.scalar_type(), "masses", ([&]{
        _MassKernel<scalar_t><<< blk, threads >>>(
                masses.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                mtop_l, step_t, mw_l, step_w, (int)steps
        ); 
    })); 
}

torch::Tensor _create_event_nlep(torch::Tensor edge_idx, torch::Tensor pid)
{
    const unsigned int threads = 1024; 
    const unsigned int nodes = pid.size(0); 
    torch::Tensor maps = torch::zeros({nodes, nodes}, _MakeOp(pid)); 
    const dim3 blk = BLOCKS(threads, nodes, nodes);  
    AT_DISPATCH_ALL_TYPES(edge_idx.scalar_type(), "n-leps", ([&]{
        _lep_map<scalar_t><<< blk, threads >>>(
                maps.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                 pid.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            edge_idx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                nodes
        ); 
    })); 
    maps = maps.sum({-1}, true); 
    return torch::cat({pid, maps}, -1); 
}

std::vector<torch::Tensor> _create_mapping(torch::Tensor edge_idx, torch::Tensor pid, torch::Tensor batch)
{
    const unsigned int threads = 1024; 
    const unsigned int idx_s = edge_idx.size(1);
    const dim3 blk = BLOCKS(threads, idx_s, idx_s);
    torch::Tensor llbb_nu = torch::zeros({idx_s*idx_s, 6}, _MakeOp(pid)); 
    batch = batch.to(_MakeOp(pid)); 

    AT_DISPATCH_ALL_TYPES(pid.scalar_type(), "mapping", ([&]{
        _MappingKernel<scalar_t><<<blk, threads>>>(
                 llbb_nu.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                   batch.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(), 
                edge_idx.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                     pid.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                    idx_s
        ); 
    })); 

    torch::Tensor msk_nu    = llbb_nu.index({torch::indexing::Slice(), 4}) >= 1; 
    torch::Tensor msk_nunu  = llbb_nu.index({torch::indexing::Slice(), 5}) >= 1; 
    torch::Tensor pair_nu   = llbb_nu.index({msk_nu}); 
    torch::Tensor pair_nunu = llbb_nu.index({msk_nunu}); 
    return {pair_nu, pair_nunu}; 
}

std::vector<torch::Tensor> _viable_solutions(
        torch::Tensor pairs, torch::Tensor mass_matrix, 
        torch::Tensor pmc, torch::Tensor t_met_xy, torch::Tensor batch,
        const double null)
{
    const unsigned int threads = 1024;
    const unsigned int lx = pairs.size(0); 
    const unsigned int ly = mass_matrix.size(0); 
    const unsigned int lz = 4*4 + 3;  
    const unsigned int len_i = ly*ly; 
    torch::Tensor met_xy = t_met_xy.index({batch});  

    torch::Tensor pmc_b1 = torch::zeros({lx, len_i, 4}, _MakeOp(pmc));
    torch::Tensor pmc_b2 = torch::zeros({lx, len_i, 4}, _MakeOp(pmc));

    torch::Tensor pmc_l1 = torch::zeros({lx, len_i, 4}, _MakeOp(pmc));
    torch::Tensor pmc_l2 = torch::zeros({lx, len_i, 4}, _MakeOp(pmc));
    torch::Tensor pair_i = torch::zeros({lx, len_i, 4}, _MakeOp(pairs)); 

    torch::Tensor mass_m1 = torch::zeros({lx, len_i, 3}, _MakeOp(pmc));
    torch::Tensor mass_m2 = torch::zeros({lx, len_i, 3}, _MakeOp(pmc));
    torch::Tensor met_xy_ = torch::zeros({lx, len_i, 2}, _MakeOp(pmc)); 

    const dim3 blk = BLOCKS(threads, lx, len_i, lz); 
    AT_DISPATCH_ALL_TYPES(pmc.scalar_type(), "assigns", ([&]{
        _assignment_kernel<scalar_t><<<blk, threads>>>(
                 pmc_b1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                 pmc_b2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

                 pmc_l1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                 pmc_l2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

                 pair_i.packed_accessor64<long    , 3, torch::RestrictPtrTraits>(),
                mass_m1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                mass_m2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                met_xy_.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 

                 met_xy.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            mass_matrix.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                    pmc.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                  pairs.packed_accessor64<long    , 2, torch::RestrictPtrTraits>(),
                    lx, ly, lz
        );
    })); 

    torch::Tensor dia_sol_o = torch::ones({lx, len_i}, _MakeOp(mass_m1))*-1; 
    torch::Tensor nu_1 = torch::zeros({lx, len_i, 4}, _MakeOp(mass_m1)); 
    torch::Tensor nu_2 = torch::zeros({lx, len_i, 4}, _MakeOp(mass_m2)); 

    const dim3 blk_ = BLOCKS(threads, ly, ly, 5); 
    for (unsigned int x(0); x < lx; ++x){ 
        std::map<std::string, torch::Tensor> nus;
        nus = _NuNu(pmc_b1[x], pmc_b2[x], pmc_l1[x], pmc_l2[x], mass_m1[x], mass_m2[x], met_xy_[x], null); 
        torch::Tensor nu1 = nus["NuVec_1"]; 
        torch::Tensor nu2 = nus["NuVec_2"]; 
        torch::Tensor dia = nus["diagonal"]; 
        torch::Tensor noS = nus["NoSols"]; 
        AT_DISPATCH_ALL_TYPES(pmc.scalar_type(), "builder", ([&]{
            _builder_nunu<scalar_t><<<blk_, threads>>>(
               dia_sol_o.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
                    nu_1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(), 
                    nu_2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),

                    noS.packed_accessor64<bool    , 1, torch::RestrictPtrTraits>(),
                    dia.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    nu1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    nu2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    x, ly, 5); 
        })); 
    }

    torch::Tensor dia_min_comb = -1*torch::ones({t_met_xy.size(0), len_i}, _MakeOp(dia_sol_o));  
    torch::Tensor dia_min_mass = -1*torch::ones({t_met_xy.size(0), lx   }, _MakeOp(dia_sol_o)); 
    const dim3 blk_x = BLOCKS(threads, lx*len_i); 
    AT_DISPATCH_ALL_TYPES(pmc.scalar_type(), "finder", ([&]{
        _min_finder<scalar_t><<<blk_x, threads>>>(
                dia_min_mass.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                dia_min_comb.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                      pair_i.packed_accessor64<long    , 3, torch::RestrictPtrTraits>(),
                       batch.packed_accessor64<long    , 1, torch::RestrictPtrTraits>(),
                   dia_sol_o.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    lx, len_i);  
    })); 

    torch::Tensor optimal = torch::cat({dia_min_comb, dia_min_mass}, -1); 
    optimal.index_put_({optimal < 0}, 1); 

    torch::Tensor min_ = std::get<0>(optimal.min({-1})); 
    torch::Tensor nu_1f = torch::zeros({t_met_xy.size(0), 4}, _MakeOp(pmc)); 
    torch::Tensor ms_1f = torch::zeros({t_met_xy.size(0), 3}, _MakeOp(pmc)); 

    torch::Tensor nu_2f = torch::zeros({t_met_xy.size(0), 4}, _MakeOp(pmc)); 
    torch::Tensor ms_2f = torch::zeros({t_met_xy.size(0), 3}, _MakeOp(pmc)); 
  
    torch::Tensor combi = torch::zeros({t_met_xy.size(0), 4}, _MakeOp(pair_i)); 

    const dim3 blk_o = BLOCKS(threads, lx, len_i, 4); 
    AT_DISPATCH_ALL_TYPES(pmc.scalar_type(), "populate", ([&]{
        _populate<scalar_t><<<blk_o, threads>>>(
            nu_1f.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            ms_1f.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            nu_2f.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            ms_2f.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            combi.packed_accessor64<long    , 2, torch::RestrictPtrTraits>(),

            nu_1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
         mass_m1.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            nu_2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
         mass_m2.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
          pair_i.packed_accessor64<long    , 3, torch::RestrictPtrTraits>(),

       dia_sol_o.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
            min_.packed_accessor64<scalar_t, 1, torch::RestrictPtrTraits>(),
           batch.packed_accessor64<long    , 1, torch::RestrictPtrTraits>(),
           lx, ly, 4); 
    })); 
    return {nu_1f, nu_2f, ms_1f, ms_2f, combi, min_.view({-1, 1})}; 
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


std::map<std::string, torch::Tensor> _NuNu(
                torch::Tensor pmc_b1, torch::Tensor pmc_b2, 
                torch::Tensor pmc_l1, torch::Tensor pmc_l2,
                torch::Tensor met_xy, torch::Tensor masses, 
                const double null)
{
    return _NuNu(pmc_b1, pmc_b2, pmc_l1, pmc_l2, masses, masses, met_xy, null); 
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


std::map<std::string, torch::Tensor> _CombinatorialCartesian(
        torch::Tensor edge_index, torch::Tensor batch, 
        torch::Tensor pmc, torch::Tensor pid, torch::Tensor met_xy, 
        const double mass_top_l, const double mass_top_u, const double mass_w_l, const double mass_w_u, 
        const double mass_nu, const double null)
{
    int steps = 50;
    torch::Tensor mass_matrix = torch::zeros({steps, 3}, _MakeOp(met_xy)); 
    _MassMatrix(mass_top_l, mass_top_u, mass_w_l, mass_w_u, mass_matrix, steps); 

    pid = pid.to(_MakeOp(edge_index)); 
    pid = _create_event_nlep(edge_index, pid); 
    std::vector<torch::Tensor> nus = _create_mapping(edge_index, pid, batch); 
    torch::Tensor pair_nu   = nus[0]; 
    torch::Tensor pair_nunu = nus[1]; 

    std::vector<torch::Tensor> res; 
    if (!pair_nunu.size(0)){
        torch::Tensor nu_null = torch::cat({torch::zeros_like(met_xy), torch::zeros_like(met_xy)}, -1); 
        res = {nu_null, nu_null, nu_null, nu_null, nu_null, nu_null}; 
    }
    else {res = _viable_solutions(pair_nunu, mass_matrix, pmc, met_xy, batch, null);}

    return {
        {"nu_1f", res[0]}, {"nu_2f", res[1]}, 
        {"ms_1f", res[2]}, {"ms_2f", res[3]}, 
        {"combi", res[4]}, {"min"  , res[5]}
    }; 
}

#endif
