#include <transform/polar-tensors/polar.h>
#include <operators.h>
#include <physics.h>
#include "nusol.h"

torch::TensorOptions Tooling::MakeOp(torch::Tensor x)
{
    torch::TensorOptions op = torch::TensorOptions().device(x.device()).dtype(x.dtype()); 
    return op; 
}

torch::Tensor Tooling::Pi_2(torch::Tensor x)
{
    torch::TensorOptions op = MakeOp(x); 
    const unsigned int dim_i = x.size(0); 
    torch::Tensor z = torch::zeros({dim_i, 1}, op); 
    return torch::acos(z); 
}

torch::Tensor Tooling::x0(torch::Tensor pmc, torch::Tensor _pm2, torch::Tensor mH2, torch::Tensor mL2)
{
    torch::Tensor e = pmc.index({torch::indexing::Slice(), 3}).view({-1, 1}); 
    return -(mH2 - mL2 - _pm2)/(2*e); 
}

torch::Tensor Tooling::Sigma(torch::Tensor x, torch::Tensor sigma)
{
    sigma = sigma.view({-1, 2, 2}); 
    const unsigned int dim_i = x.size(0);
    const unsigned int dim_i_ = sigma.size(0); 
    const torch::TensorOptions op = Tooling::MakeOp(x); 
    if (dim_i != dim_i_)
    {
        torch::Tensor tmp = torch::ones({dim_i, 2, 2}, op); 
        sigma = tmp * sigma[0]; 
    }
    sigma = torch::inverse(sigma); 
    sigma = torch::pad(sigma, {0, 1, 0, 1}, "constant", 0);
    sigma = torch::transpose(sigma, 1, 2);
    return sigma; 
}

torch::Tensor Tooling::MET(torch::Tensor met_xy)
{
    torch::Tensor matrix = met_xy.view({-1, 2});
    matrix = torch::pad(matrix, {0, 1, 0, 0}, "constant", 0).view({-1, 3, 1}); 
    torch::Tensor t0 = torch::zeros_like(matrix); 
    return torch::cat({t0, t0, matrix}, -1); 
}

std::map<std::string, torch::Tensor> Tooling::GetMasses(torch::Tensor L, torch::Tensor masses)
{
    const unsigned int dim_i = L.size(0);
    masses = masses.view({-1, 3}); 
    const unsigned int dim_i_ = masses.size(0); 
    if (dim_i != dim_i_)
    {
        torch::Tensor tmp = torch::ones({dim_i, 3}, MakeOp(masses)); 
        masses = tmp*(masses[0]); 
    }
    std::map<std::string, torch::Tensor> out; 
    out["W2"] = torch::pow(masses.index({torch::indexing::Slice(), 0}), 2).view({dim_i, 1});  
    out["T2"] = torch::pow(masses.index({torch::indexing::Slice(), 1}), 2).view({dim_i, 1});  
    out["N2"] = torch::pow(masses.index({torch::indexing::Slice(), 2}), 2).view({dim_i, 1});  
    return out; 
}

torch::Tensor Tooling::Rotation(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor base)
{
    torch::Tensor pmc_b3  = pmc_b.index( {torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 
    torch::Tensor pmc_mu3 = pmc_mu.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 

    torch::Tensor muphi = Transform::Tensors::Phi(pmc_mu); 
    torch::Tensor theta = Physics::Tensors::Theta(pmc_mu); 

    torch::Tensor Rz = Operators::Tensors::Rz(-muphi); 
    torch::Tensor Ry = Operators::Tensors::Ry(Tooling::Pi_2(theta) - theta); 

    torch::Tensor Rx = torch::matmul(Rz, pmc_b3.view({-1, 3, 1}));
    Rx = torch::matmul(Ry, Rx.view({-1, 3, 1})); 
    Rx = -torch::atan2(
            Rx.index({torch::indexing::Slice(), 2}), 
            Rx.index({torch::indexing::Slice(), 1})
    ).view({-1, 1}); 

    Rx = Operators::Tensors::Rx(Rx); 
    Rx = torch::transpose(Rx, 1, 2); 
    Ry = torch::transpose(Ry, 1, 2); 
    Rz = torch::transpose(Rz, 1, 2); 

    return torch::matmul(torch::matmul(Rz, torch::matmul(Ry, Rx)), base); 
}

torch::Tensor Tooling::Shape(torch::Tensor x, std::vector<int> diag)
{
    const torch::TensorOptions op = Tooling::MakeOp(x);
    torch::Tensor shape = torch::diag(torch::tensor(diag, op)).view({-1, 3, 3}); 
    torch::Tensor ones = torch::ones({x.size(0), 3, 3}, op); 
    shape = (ones*shape).view({-1, 3, 3}); 
    return shape; 
}

torch::Tensor Tooling::HorizontalVertical(torch::Tensor G)
{
    torch::Tensor G01 = G.index({torch::indexing::Slice(), 0, 1}).view({-1, 1, 1}); 
    torch::Tensor G02 = G.index({torch::indexing::Slice(), 0, 2}).view({-1, 1, 1}); 
    torch::Tensor G12 = G.index({torch::indexing::Slice(), 1, 2}).view({-1, 1, 1}); 
    torch::Tensor t0 = zeros_like(G01); 
    return torch::cat({ G01, t0, G12, t0, G01, G02 - G12, t0, t0, t0}, -1).view({-1, 3, 3}); 
}

torch::Tensor Tooling::Parallel(torch::Tensor G, torch::Tensor CoF)
{
    torch::Tensor g00 = -CoF.index({torch::indexing::Slice(), 0, 0}); 
    torch::Tensor G00 = G.index({torch::indexing::Slice(), 0, 0}); 
    torch::Tensor G01 = G.index({torch::indexing::Slice(), 0, 1}); 
    torch::Tensor G11 = G.index({torch::indexing::Slice(), 1, 1});
    torch::Tensor G12 = G.index({torch::indexing::Slice(), 1, 2});
    torch::Tensor t0 = torch::zeros_like(G12); 
    torch::Tensor G_ = torch::zeros_like(G); 
    std::vector<signed long> dims = {-1, 1, 1};  

    // ----- Populate cases where g00 is 0
    torch::Tensor y0 = g00 == 0.; 	
    G_.index_put_({y0}, torch::cat({
                G01.index({y0}).view(dims), G11.index({y0}).view(dims), G12.index({y0}).view(dims), 
                 t0.index({y0}).view(dims),  t0.index({y0}).view(dims),  t0.index({y0}).view(dims), 
                 t0.index({y0}).view(dims),  t0.index({y0}).view(dims),  t0.index({y0}).view(dims)
    }, -1).view({-1, 3, 3})); 
    
    // ----- Populate cases where -g00 > 0
    torch::Tensor yr = g00 > 0.; 
    G_.index_put_({yr}, torch::cat({
                G01.index({yr}).view(dims), 
                G11.index({yr}).view(dims), 
                (G12.index({yr}) - torch::sqrt(g00.index({yr}))).view(dims),
                
                G01.index({yr}).view(dims), 
                G11.index({yr}).view(dims), 
                (G12.index({yr}) + torch::sqrt(g00.index({yr}))).view(dims),
                
                torch::cat({t0.index({yr}), t0.index({yr}), t0.index({yr})}, -1).view({-1, 1, 3})
    }, -1).view({-1, 3, 3})); 
    return G_; 
}

torch::Tensor Tooling::Intersecting(torch::Tensor G, torch::Tensor g22, torch::Tensor CoF)
{
    std::vector<signed long> dim11 = {-1, 1, 1}; 
    std::vector<signed long> dim13 = {-1, 1, 3};

    torch::Tensor g02 = CoF.index({torch::indexing::Slice(), 0, 2}); 
    torch::Tensor g12 = CoF.index({torch::indexing::Slice(), 1, 2}); 
    
    torch::Tensor x0 = (g02 / g22); 
    torch::Tensor y0 = (g12 / g22); 
    torch::Tensor G11 = G.index({torch::indexing::Slice(), 1, 1}); 
    torch::Tensor G01 = G.index({torch::indexing::Slice(), 0, 1}); 
    torch::Tensor t0 = torch::zeros_like(G01); 

    // Case 1: -g22 < 0 - No Solution  - ignore
    torch::Tensor G_ = torch::zeros_like(G);

    // Case 2: -g22 == 0 - One Solution 
    torch::Tensor _s1 = -g22 == 0.; 
    torch::Tensor _s1_s = _s1.sum(-1);  
    
    // Case 3; -g22 > 0 - Two Solutions 
    torch::Tensor _s2 = -g22 > 0.;  
    torch::Tensor _s2_s = _s2.sum(-1);

    if (_s1_s.item<int>() > 0)
    {
        // ---------- Case 2 ----------- //   
        G_.index_put_({_s1}, torch::cat({
                G01.index({_s1}).view(dim11), 
                G11.index({_s1}).view(dim11), 
                ((-G11.index({_s1}) * y0.index({_s1})) - (G01.index({_s1}) * x0.index({_s1}))).view(dim11), 

                torch::cat({t0.index({_s1}), t0.index({_s1}), t0.index({_s1})}, -1).view(dim13), 
                torch::cat({t0.index({_s1}), t0.index({_s1}), t0.index({_s1})}, -1).view(dim13)
        }, -1).view({-1, 3, 3})); 
    }
    if (_s2_s.item<int>() > 0)
    {
        // ---------- Case 3 ----------- //   
        torch::Tensor _s = torch::sqrt(-g22.index({_s2})); 
        G_.index_put_({_s2}, torch::cat({
                // Solution 1
                (G01.index({_s2}) - _s).view(dim11), 
                (G11.index({_s2})).view(dim11), 
                (-G11.index({_s2}) * y0.index({_s2}) - (G01.index({_s2}) - _s) * x0.index({_s2})).view(dim11), 

                // Solution 2
                (G01.index({_s2}) + _s).view(dim11), 
                (G11.index({_s2})).view(dim11), 
                (-G11.index({_s2}) * y0.index({_s2}) - (G01.index({_s2}) + _s) * x0.index({_s2})).view(dim11), 
                torch::cat({t0.index({_s2}), t0.index({_s2}), t0.index({_s2})}, -1).view(dim13) // Zeros 
        }, -1).view({-1, 3, 3})); 
    }
    return G_; 
}

torch::Tensor NuSol::Tensor::BaseMatrix(torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor masses)
{
    pmc_b  = pmc_b.view( {-1, 4}); 
    pmc_mu = pmc_mu.view({-1, 4}); 
    std::map<std::string, torch::Tensor> mass = Tooling::GetMasses(pmc_b, masses); 
    const torch::TensorOptions op = Tooling::MakeOp(pmc_b); 
    const unsigned int dim_i = pmc_b.size(0); 

    torch::Tensor bB   = Physics::Tensors::Beta(pmc_b); 
    torch::Tensor muB  = Physics::Tensors::Beta(pmc_mu); 
    torch::Tensor muB2 = Physics::Tensors::Beta2(pmc_mu); 
    torch::Tensor muP  = Physics::Tensors::P(pmc_mu); 

    torch::Tensor x0p = Tooling::x0(pmc_b , Physics::Tensors::M2(pmc_b ), mass["T2"], mass["W2"]); 
    torch::Tensor x0  = Tooling::x0(pmc_mu, Physics::Tensors::M2(pmc_mu), mass["W2"], mass["N2"]); 
  
    torch::Tensor pmc_b3  = pmc_b.index( {torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 
    torch::Tensor pmc_mu3 = pmc_mu.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 

    torch::Tensor c_ = Operators::Tensors::CosTheta(pmc_b3, pmc_mu3); 
    torch::Tensor s_ = Operators::Tensors::SinTheta(pmc_b3, pmc_mu3);
    
    torch::Tensor tmp_ = muB / bB; 
    torch::Tensor w_ = ( - tmp_ - c_ ) / s_; 
    torch::Tensor w = ( tmp_ - c_ ) / s_; 
    
    torch::Tensor O2 = w.pow(2) + 1 - muB2;
    torch::Tensor e2 = (mass["W2"] - mass["N2"]) * ( 1 - muB2 ); 
    
    torch::Tensor Sx = (x0 * muB - muP * ( 1 - muB2 )) / muB2; 
    torch::Tensor Sy = ( (x0p / bB) - c_ * Sx ) / s_; 
    
    tmp_ = Sx + w*Sy; 
    torch::Tensor x1 = Sx - tmp_ / O2; 
    torch::Tensor y1 = Sy - tmp_ * (w / O2); 
    torch::Tensor Z = torch::sqrt(torch::relu(x1.pow(2) * O2 - ( Sy - w*Sx ).pow(2) - ( mass["W2"] - x0.pow(2) - e2 ))); 

    torch::Tensor O = torch::sqrt(O2);
    torch::Tensor _0 = torch::zeros({dim_i, 1}, op);
    return torch::cat({ Z / O, _0, x1 - muP, (w*Z)/O, _0, y1, _0, Z, _0}, -1).view({dim_i, 3, 3}); 
}

std::tuple<torch::Tensor, torch::Tensor> NuSol::Tensor::Intersection(torch::Tensor A, torch::Tensor B, const double null)
{
    const torch::TensorOptions op = Tooling::MakeOp(A);
    const unsigned int dim_i = A.size(0);
    std::vector<signed long> dim11 = {-1, 1, 1}; 
    std::vector<signed long> dim13 = {-1, 1, 3}; 
    std::vector<signed long> dim31 = {-1, 3, 1}; 
    std::vector<signed long> dim313 = {-1, 3, 1, 3}; 
    std::vector<signed long> dim133 = {-1, 1, 3, 3}; 
    std::vector<signed long> dim331 = {-1, 3, 3, 1}; 
    std::vector<signed long> dim333 = {-1, 3, 3, 3}; 
    A = A.clone(); 
    B = B.clone(); 

    torch::Tensor detA = torch::abs(torch::det(A)); 
    torch::Tensor detB = torch::abs(torch::det(B)); 

    // Perform the variable swaps 
    torch::Tensor swp = detB > detA;
    torch::Tensor _tmp = B.index({swp}); 
    B.index_put_({swp}, A.index({swp})); 
    A.index_put_({swp}, _tmp);
    
    // Find the non imaginary eigenvalue solutions
    _tmp = torch::linalg::eigvals(torch::inverse(A).matmul(B)); 
    torch::Tensor _r = torch::real(_tmp); 
    torch::Tensor msk = torch::isreal(_tmp)*torch::arange(3, 0, -1, op); 
    msk = torch::argmax(msk, -1, true); 
    _r = torch::gather(_r, 1, msk).view({-1, 1, 1}); 
    torch::Tensor G = B - _r*A;
    
    // Get the diagonals of the matrix
    torch::Tensor G00 = G.index({torch::indexing::Slice(), 0, 0}); 
    torch::Tensor G11 = G.index({torch::indexing::Slice(), 1, 1}); 
    
    // ----- Numerical Stability part ----- // 
    swp = torch::abs(G00) > torch::abs(G11); 
    G.index_put_({swp}, torch::cat({
    			G.index({swp, 1, torch::indexing::Slice()}).view(dim13), 
    			G.index({swp, 0, torch::indexing::Slice()}).view(dim13), 
    			G.index({swp, 2, torch::indexing::Slice()}).view(dim13)}, 1)); 

    G.index_put_({swp}, torch::cat({
    			G.index({swp, torch::indexing::Slice(), 1}).view(dim31), 
    			G.index({swp, torch::indexing::Slice(), 0}).view(dim31), 
    			G.index({swp, torch::indexing::Slice(), 2}).view(dim31)}, 2)); 

    G = G/(G.index({torch::indexing::Slice(), 1, 1}).view({-1, 1, 1})); 
    
    torch::Tensor CoF = Operators::Tensors::CoFactors(G);
    torch::Tensor g22 = CoF.index({torch::indexing::Slice(), 2, 2}); 
    torch::Tensor _out = zeros_like(G); 
    
    detA = detA == 0; 
    torch::Tensor G00_G11 = ( G00 == 0 ) * (G11 == 0); 
    torch::Tensor g22__   = (-g22 <= 0.) * (G00_G11 == false); 	
    torch::Tensor g22_    = (-g22 >  0.) * (G00_G11 == false); 

    // 1. ----- Case where the solutions are Horizontal and Vertical ----- //
    torch::Tensor SolG_HV = Tooling::HorizontalVertical(G.index({G00_G11}));
    
    // 2. ------ Case where the solutions are parallel 
    torch::Tensor SolG_Para = Tooling::Parallel(G.index({g22__}), CoF.index({g22__})); 
    
    // 3. ------- Case where the solutions are intersecting 
    torch::Tensor SolG_Int = Tooling::Intersecting(G.index({g22_}), g22.index({g22_}), CoF.index({g22_})); 

    _out.index_put_({G00_G11}, SolG_HV); 
    _out.index_put_({g22__  }, SolG_Para);
    _out.index_put_({g22_   }, SolG_Int);
    _out.index_put_({detA   }, 0); 
    
    // Swap the XY if they were swapped previously 	
    _tmp = torch::cat({
    		_out.index({swp, torch::indexing::Slice(), 1}).view(dim31), 
    		_out.index({swp, torch::indexing::Slice(), 0}).view(dim31), 
    		_out.index({swp, torch::indexing::Slice(), 2}).view(dim31)}, 2);
    _out.index_put_({swp}, _tmp);

    // ------ Intersection of line with Ellipse ------ //
    torch::Tensor _t, d1, V, V_, diag; 
    V = torch::cross(_out.view(dim313), A.view(dim133), 3); 
    V = torch::transpose(V, 2, 3); 
    V = std::get<1>(torch::linalg::eig(V));
    V = torch::real(V);
    V = torch::transpose(V, 2, 3); 
    
    _t = V / (V.index({
                torch::indexing::Slice(), 
    		torch::indexing::Slice(), 
    		torch::indexing::Slice(), 2}).view(dim331)); 
    
    d1 = torch::sum(((_out.view(dim313))*V), 3).pow(2);
    V_ = torch::reshape(V, dim333); 
    
    // ------- Neutrino Solution ------ //
    _t = _t.view({dim_i, -1, 3});  
    _t = torch::nan_to_num(_t, 0, 0);

    diag = torch::matmul(V_, A.view(dim133)); 
    diag = torch::sum((diag * V_), {-1}).pow(2); 
    diag = (d1 + diag).view({dim_i, -1});   
    
    // ------- Remove false solutions ------ //
    torch::Tensor t0_, t1_, t2_;  
    t0_ = _t.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).view({dim_i, -1});  
    t1_ = _t.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).view({dim_i, -1});     
    t2_ = _t.index({torch::indexing::Slice(), torch::indexing::Slice(), 2}).view({dim_i, -1}); 
    detB = ((t0_.view({dim_i, -1, 1}) == 0) * (t1_.view({dim_i, -1, 1}) == 0)).view({dim_i, -1}); 

    t2_.index_put_({detB}, 0);
    diag.index_put_({detA}, -1); 
    diag.index_put_({detB}, -1); 
    diag.index_put_({diag > null}, -1); 
    
    // ------- Do the Sorting ------ //
    torch::Tensor id = std::get<1>(diag.sort(-1, false)); 
    t0_ = torch::gather(t0_, -1, id).view({dim_i, -1, 1}); 
    t1_ = torch::gather(t1_, -1, id).view({dim_i, -1, 1}); 
    t2_ = torch::gather(t2_, -1, id).view({dim_i, -1, 1}); 
    _t = torch::cat({t0_, t1_, t2_}, -1); 
    diag = torch::gather(diag, 1, id);
   
    int max_len = diag.size(-1) - torch::max((diag != -1).sum(-1)).item<int>(); 
    _t = _t.index({
            torch::indexing::Slice(), 
            torch::indexing::Slice(max_len, torch::indexing::None), 
            torch::indexing::Slice()}); 

    diag = diag.index({
            torch::indexing::Slice(), 
            torch::indexing::Slice(max_len, torch::indexing::None)}); 

    return {_t, diag};  
}

std::map<std::string, torch::Tensor> NuSol::Tensor::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, 
        torch::Tensor met_xy, torch::Tensor masses, torch::Tensor sigma)
{
    
    // Base matrix - the Analytical solutions and constants
    torch::Tensor base = NuSol::Tensor::BaseMatrix(pmc_b, pmc_mu, masses); 
    const torch::Tensor H_base = Tooling::Rotation(pmc_b, pmc_mu, base);
    const torch::Tensor Pi2 = Tooling::Pi_2(base); 
    torch::Tensor Derivative = Operators::Tensors::Rz(Pi2); 
    Derivative = Derivative.matmul(Tooling::Shape(Pi2, {1, 1, 0})); 

    // Invert the input sigma matrix 
    sigma = Tooling::Sigma(met_xy, sigma); 
   
    // ------- Convert missing energy matrix into tensor ----- //
    met_xy = Tooling::MET(met_xy); 

    // ------- create ellipse ------- //
    torch::Tensor dNu, X_, M_; 
    dNu = met_xy - H_base; 
    X_ = torch::matmul(torch::transpose(dNu, 1, 2), sigma); 
    X_ = torch::matmul(X_, dNu).view({-1, 3, 3}); 
    M_ = X_.matmul(Derivative); 
    M_ = M_ + torch::transpose(M_, 1, 2);  
    
    std::map<std::string, torch::Tensor> output; 
    output["M"] = M_;
    output["H"] = H_base; 
    output["X"] = X_; 
    return output; 
}

std::map<std::string, torch::Tensor> NuSol::Tensor::Nu(
        torch::Tensor pmc_b, torch::Tensor pmc_mu, torch::Tensor met_xy, 
        torch::Tensor masses, torch::Tensor sigma, const double null)
{
    torch::Tensor v, chi2; 
    std::map<std::string, torch::Tensor> _Nu, output; 
    std::tuple<torch::Tensor, torch::Tensor> sols; 
    const unsigned int dim_i = met_xy.size(0); 

    _Nu = NuSol::Tensor::Nu(pmc_b, pmc_mu, met_xy, masses, sigma); 
    torch::Tensor M = _Nu["M"]; 
    torch::Tensor H = _Nu["H"]; 
    torch::Tensor X = _Nu["X"]; 

    sols = NuSol::Tensor::Intersection(M, Tooling::Shape(met_xy, {1, 1, -1}), null); 
    v = std::get<0>(sols).view({dim_i, -1, 1, 3}); 

    torch::Tensor diag = std::get<1>(sols); 
    chi2 = (v * X.view({dim_i, 1, 3, 3})).sum({-1}); 
    chi2 = chi2.view({dim_i, -1, 3})*v.view({dim_i, -1, 3}); 
    chi2 = chi2.sum(-1); 

    v = (H.view({dim_i, 1, 3, 3})*v).sum(-1);
    
    torch::Tensor t0_, t1_, t2_;  
    t0_ = v.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).view({dim_i, -1});  
    t1_ = v.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).view({dim_i, -1});     
    t2_ = v.index({torch::indexing::Slice(), torch::indexing::Slice(), 2}).view({dim_i, -1}); 

    // ------- Do the Sorting ------ //
    torch::Tensor id = std::get<1>(chi2.sort(-1, false)); 
    t0_ = torch::gather(t0_, -1, id).view({dim_i, -1, 1}); 
    t1_ = torch::gather(t1_, -1, id).view({dim_i, -1, 1}); 
    t2_ = torch::gather(t2_, -1, id).view({dim_i, -1, 1}); 
    v = torch::cat({t0_, t1_, t2_}, -1); 
    chi2 = torch::gather(diag, 1, id);

    output["NuVec"] = v; 
    output["chi2"] = chi2;

    return output; 
}

std::map<std::string, torch::Tensor> NuSol::Tensor::NuNu(
        torch::Tensor pmc_b1, torch::Tensor pmc_b2, torch::Tensor pmc_l1, torch::Tensor pmc_l2, 
        torch::Tensor met_x , torch::Tensor met_y, torch::Tensor masses, const double null)
{
    std::map<std::string, torch::Tensor> output; 

    return output; 
}
