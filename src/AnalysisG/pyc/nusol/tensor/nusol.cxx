#include <operators/operators.h>
#include <transform/transform.h>
#include <physics/physics.h>
#include <utils/utils.h>
#include <nusol/nusol.h>


std::map<std::string, torch::Tensor> GetMasses(torch::Tensor* L, torch::Tensor* masses){
    torch::Tensor _masses = masses -> view({-1, 3}); 
    const unsigned int dim_i = L -> size(0);
    const unsigned int dim_i_ = _masses.size(0); 
    if (dim_i != dim_i_){_masses = torch::ones({dim_i, 3}, MakeOp(masses))*_masses[0];}
    std::map<std::string, torch::Tensor> out; 
    out["T2"] = torch::pow(_masses.index({torch::indexing::Slice(), 0}), 2).view({dim_i, 1});  
    out["W2"] = torch::pow(_masses.index({torch::indexing::Slice(), 1}), 2).view({dim_i, 1});  
    out["N2"] = torch::pow(_masses.index({torch::indexing::Slice(), 2}), 2).view({dim_i, 1});  
    return out; 
}

torch::Tensor _x0(torch::Tensor* pmc, torch::Tensor* _pm2, torch::Tensor* mH2, torch::Tensor* mL2){
    torch::Tensor e = pmc -> index({torch::indexing::Slice(), 3}).view({-1, 1}); 
    return -(*mH2 - *mL2 - *_pm2)/(2*e); 
}

torch::Tensor HorizontalVertical(torch::Tensor* G){
    torch::Tensor G01 = G -> index({torch::indexing::Slice(), 0, 1}).view({-1, 1, 1}); 
    torch::Tensor G02 = G -> index({torch::indexing::Slice(), 0, 2}).view({-1, 1, 1}); 
    torch::Tensor G12 = G -> index({torch::indexing::Slice(), 1, 2}).view({-1, 1, 1}); 
    torch::Tensor t0 = zeros_like(G01); 
    return torch::cat({ G01, t0, G12, t0, G01, G02 - G12, t0, t0, t0}, -1).view({-1, 3, 3}); 
}

torch::Tensor Parallel(torch::Tensor* G, torch::Tensor* CoF){
    torch::Tensor g00 = -CoF -> index({torch::indexing::Slice(), 0, 0}); 
    torch::Tensor G00 = G -> index({torch::indexing::Slice(), 0, 0}); 
    torch::Tensor G01 = G -> index({torch::indexing::Slice(), 0, 1}); 
    torch::Tensor G11 = G -> index({torch::indexing::Slice(), 1, 1});
    torch::Tensor G12 = G -> index({torch::indexing::Slice(), 1, 2});
    torch::Tensor t0 = torch::zeros_like(G12); 
    torch::Tensor G_ = torch::zeros_like(*G); 
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

torch::Tensor Intersecting(torch::Tensor* G, torch::Tensor* g22, torch::Tensor* CoF){
    std::vector<signed long> dim11 = {-1, 1, 1}; 
    std::vector<signed long> dim13 = {-1, 1, 3};

    torch::Tensor g02 = CoF -> index({torch::indexing::Slice(), 0, 2}); 
    torch::Tensor g12 = CoF -> index({torch::indexing::Slice(), 1, 2}); 
    
    torch::Tensor x0 = (g02 / *g22); 
    torch::Tensor y0 = (g12 / *g22); 
    torch::Tensor G11 = G -> index({torch::indexing::Slice(), 1, 1}); 
    torch::Tensor G01 = G -> index({torch::indexing::Slice(), 0, 1}); 
    torch::Tensor t0 = torch::zeros_like(G01); 

    // Case 1: -g22 < 0 - No Solution  - ignore
    torch::Tensor G_ = torch::zeros_like(*G);

    // Case 2: -g22 == 0 - One Solution 
    torch::Tensor _s1 = -(*g22) == 0.; 
    torch::Tensor _s1_s = _s1.sum(-1);  
    
    // Case 3; -g22 > 0 - Two Solutions 
    torch::Tensor _s2 = -(*g22) > 0.;  
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
        torch::Tensor _s = torch::sqrt(-g22 -> index({_s2})); 
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

torch::Tensor Rotation(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* base){
    torch::Tensor pmc_b3  = pmc_b -> index( {torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 
    torch::Tensor pmc_mu3 = pmc_mu -> index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 

    torch::Tensor muphi = -transform_::Phi(pmc_mu); 
    torch::Tensor theta = physics_::Theta(pmc_mu); 

    torch::Tensor Rz = operators_::Rz(&muphi); 
    torch::Tensor rx = operators_::Pi_2(&theta) - theta;
    torch::Tensor Ry = operators_::Ry(&rx); 

    torch::Tensor Rx = torch::matmul(Rz, pmc_b3.view({-1, 3, 1}));
    Rx = torch::matmul(Ry, Rx.view({-1, 3, 1})); 
    Rx = -torch::atan2(
            Rx.index({torch::indexing::Slice(), 2}), 
            Rx.index({torch::indexing::Slice(), 1})
    ).view({-1, 1}); 

    Rx = operators_::Rx(&Rx); 
    Rx = torch::transpose(Rx, 1, 2); 
    Ry = torch::transpose(Ry, 1, 2); 
    Rz = torch::transpose(Rz, 1, 2); 
    return torch::matmul(torch::matmul(Rz, torch::matmul(Ry, Rx)), *base); 
}

torch::Tensor Shape(torch::Tensor* x, std::vector<int> diag){
    const torch::TensorOptions op = MakeOp(x);
    torch::Tensor shape = torch::diag(torch::tensor(diag, op)).view({-1, 3, 3}); 
    torch::Tensor ones = torch::ones({x -> size(0), 3, 3}, op); 
    shape = (ones*shape).view({-1, 3, 3}); 
    return shape; 
}


torch::Tensor Sigma(torch::Tensor* x, torch::Tensor* sigma){
    torch::Tensor _sigma = sigma -> view({-1, 2, 2}); 
    const unsigned int dim_i = x -> size(0);
    const unsigned int dim_i_ = _sigma.size(0); 
    const torch::TensorOptions op = MakeOp(x); 
    if (dim_i != dim_i_){_sigma = torch::ones({dim_i, 2, 2}, op) * _sigma[0];}
    _sigma = torch::inverse(_sigma); 
    _sigma = torch::pad(_sigma, {0, 1, 0, 1}, "constant", 0);
    _sigma = torch::transpose(_sigma, 1, 2);
    return _sigma; 
}

torch::Tensor _met(torch::Tensor* met_xy){
    torch::Tensor matrix = met_xy -> view({-1, 2});
    matrix = torch::pad(matrix, {0, 1, 0, 0}, "constant", 0).view({-1, 3, 1}); 
    torch::Tensor t0 = torch::zeros_like(matrix); 
    return torch::cat({t0, t0, matrix}, -1); 
}

torch::Tensor _H_perp(torch::Tensor* base){
    torch::Tensor H = base -> clone(); 
    H.index_put_({torch::indexing::Slice(), 2, torch::indexing::Slice()}, 0); 
    H.index_put_({torch::indexing::Slice(), 2, 2}, 1);  
    return H; 
}

torch::Tensor _N(torch::Tensor* hperp){
    torch::Tensor H = std::get<1>(operators_::Inverse(hperp)); 
    torch::Tensor H_T = torch::transpose(H, 1, 2);    
    H_T = torch::matmul(H_T, Shape(&H_T, {1, 1, -1})); 
    return torch::matmul(H_T, H); 
}


torch::Tensor nusol_::BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses, bool null){

    torch::Tensor pmx_b  = pmc_b -> view({-1, 4}); 
    torch::Tensor pmx_mu = pmc_mu -> view({-1, 4}); 
    std::map<std::string, torch::Tensor> mass = GetMasses(&pmx_b, masses); 

    const torch::TensorOptions op = MakeOp(pmc_b); 
    const unsigned int dim_i = pmx_b.size(0); 

    torch::Tensor bB   = physics_::Beta(&pmx_b); 
    torch::Tensor muB  = physics_::Beta(&pmx_mu); 
    torch::Tensor muB2 = physics_::Beta2(&pmx_mu); 
    torch::Tensor muP  = physics_::P(&pmx_mu); 

    torch::Tensor m2b = physics_::M2(&pmx_b); 
    torch::Tensor m2l = physics_::M2(&pmx_mu); 

    torch::Tensor x0p = _x0(&pmx_b , &m2b, &mass["T2"], &mass["W2"]); 
    torch::Tensor x0  = _x0(&pmx_mu, &m2l, &mass["W2"], &mass["N2"]); 
  
    torch::Tensor pmc_b3  = pmx_b.index( {torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 
    torch::Tensor pmc_mu3 = pmx_mu.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}); 

    torch::Tensor c_ = operators_::CosTheta(&pmc_b3, &pmc_mu3); 
    torch::Tensor s_ = operators_::SinTheta(&pmc_b3, &pmc_mu3);
   
    torch::Tensor tmp_ = muB / bB; 
    torch::Tensor w_ = ( - tmp_ - c_ ) / s_; 
    torch::Tensor w  = ( tmp_ - c_ ) / s_; 
    
    torch::Tensor O2 = w.pow(2) + 1 - muB2;
    torch::Tensor e2 = (mass["W2"] - mass["N2"]) * ( 1 - muB2 );

    torch::Tensor Sx = (x0 * muB - muP * ( 1 - muB2 )) / muB2; 
    torch::Tensor Sy = ( (x0p / bB) - c_ * Sx ) / s_; 

    tmp_ = Sx + w*Sy; 
    torch::Tensor x1 = Sx - tmp_ / O2; 
    torch::Tensor y1 = Sy - tmp_ * (w / O2); 
    torch::Tensor Z = torch::sqrt(torch::relu(x1.pow(2) * O2 - ( Sy - w*Sx ).pow(2) - ( mass["W2"] - x0.pow(2) - e2 ))); 

    torch::Tensor  O = torch::sqrt(O2);
    torch::Tensor _0 = torch::zeros({dim_i, 1}, op);
    return torch::cat({ Z / O, _0, x1 - muP, (w*Z)/O, _0, y1, _0, Z, _0}, -1).view({dim_i, 3, 3}); 
}

torch::Tensor nusol_::Hperp(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses){
    torch::Tensor H = nusol_::BaseMatrix(pmc_b, pmc_mu, masses, true); 
    H = Rotation(pmc_b, pmc_mu, &H); 
    H = _H_perp(&H); 
    torch::Tensor nullx = torch::isnan(H).sum(-1).sum(-1) == 0; 
    H.index_put_({nullx, torch::indexing::Slice(), torch::indexing::Slice()}, 0); 
    return H; 
}

std::tuple<torch::Tensor, torch::Tensor> nusol_::Intersection(torch::Tensor* A, torch::Tensor* B, double null){
    const torch::TensorOptions op = MakeOp(A);
    const unsigned int dim_i = A -> size(0);
    std::vector<signed long> dim11 = {-1, 1, 1}; 
    std::vector<signed long> dim13 = {-1, 1, 3}; 
    std::vector<signed long> dim31 = {-1, 3, 1}; 
    std::vector<signed long> dim313 = {-1, 3, 1, 3}; 
    std::vector<signed long> dim133 = {-1, 1, 3, 3}; 
    std::vector<signed long> dim331 = {-1, 3, 3, 1}; 
    std::vector<signed long> dim333 = {-1, 3, 3, 3}; 
    torch::Tensor _A = A -> clone(); 
    torch::Tensor _B = B -> clone(); 

    torch::Tensor detA = torch::abs(operators_::Determinant(A)); 
    torch::Tensor detB = torch::abs(operators_::Determinant(B)); 

    // Perform the variable swaps 
    torch::Tensor swp = detB > detA;
    torch::Tensor _tmp = B -> index({swp}); 
    _B.index_put_({swp}, A -> index({swp})); 
    _A.index_put_({swp}, _tmp);
    
    // Find the non imaginary eigenvalue solutions
    _tmp = torch::linalg_eigvals(std::get<1>(operators_::Inverse(&_A)).matmul(_B)); 
    torch::Tensor _r = torch::real(_tmp); 
    torch::Tensor msk = torch::isreal(_tmp)*torch::arange(3, 0, -1, op); 
    msk = torch::argmax(msk, -1, true); 
    _r = torch::gather(_r, 1, msk).view({-1, 1, 1}); 
    torch::Tensor G = _B - _r*_A;
    
    // Get the diagonals of the matrix
    torch::Tensor G00 = G.index({torch::indexing::Slice(), 0, 0}); 
    torch::Tensor G11 = G.index({torch::indexing::Slice(), 1, 1}); 
    
    // ----- Numerical Stability part ----- // 
    swp = torch::abs(G00) > torch::abs(G11); 
    G.index_put_({swp}, torch::cat({
    			G.index({swp, 1, torch::indexing::Slice()}).view(dim13), 
    			G.index({swp, 0, torch::indexing::Slice()}).view(dim13), 
    			G.index({swp, 2, torch::indexing::Slice()}).view(dim13)
    }, 1)); 

    G.index_put_({swp}, torch::cat({
    			G.index({swp, torch::indexing::Slice(), 1}).view(dim31), 
    			G.index({swp, torch::indexing::Slice(), 0}).view(dim31), 
    			G.index({swp, torch::indexing::Slice(), 2}).view(dim31)
    }, 2)); 

    G = G/(G.index({torch::indexing::Slice(), 1, 1}).view({-1, 1, 1})); 
    
    torch::Tensor CoF = operators_::CoFactors(&G);
    torch::Tensor g22 = CoF.index({torch::indexing::Slice(), 2, 2}); 
    torch::Tensor _out = zeros_like(G); 
    
    detA = detA == 0; 
    torch::Tensor G00_G11 = ( G00 == 0 ) * (G11 == 0); 
    torch::Tensor g22__   = (-g22 <= 0.) * (G00_G11 == false); 	
    torch::Tensor g22_    = (-g22 >  0.) * (G00_G11 == false); 

    // 1. ----- Case where the solutions are Horizontal and Vertical ----- //
    torch::Tensor tx = G.index({G00_G11}); 
    torch::Tensor SolG_HV = HorizontalVertical(&tx);
    
    // 2. ------ Case where the solutions are parallel 
    torch::Tensor g22x = G.index({g22__}); 
    torch::Tensor g22f = CoF.index({g22__}); 
    torch::Tensor SolG_Para = Parallel(&g22x, &g22f); 
    
    // 3. ------- Case where the solutions are intersecting 
    g22x = G.index({g22_}); 
    g22f = CoF.index({g22_}); 
    tx   = g22.index({g22_}); 
    torch::Tensor SolG_Int = Intersecting(&g22x, &tx, &g22f); 

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
    V = torch::cross(_out.view(dim313), _A.view(dim133), 3); 
    V = torch::transpose(V, 2, 3); 
    V = std::get<1>(torch::linalg_eig(V));
    V = torch::real(V);
    V = torch::transpose(V, 2, 3); 
    
    _t = V / (V.index({
                torch::indexing::Slice(), 
    		torch::indexing::Slice(), 
    		torch::indexing::Slice(), 
    2}).view(dim331)); 
    
    d1 = torch::sum(((_out.view(dim313))*V), 3).pow(2);
    V_ = torch::reshape(V, dim333); 
    
    // ------- Neutrino Solution ------ //
    _t = _t.view({dim_i, -1, 3});  
    _t = torch::nan_to_num(_t, 0, 0);

    diag = torch::matmul(V_, _A.view(dim133)); 
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

std::map<std::string, torch::Tensor> _xNu(
        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
        torch::Tensor* masses, torch::Tensor* sigma
){
    
    // Base matrix - the Analytical solutions and constants
    torch::Tensor base = nusol_::BaseMatrix(pmc_b, pmc_mu, masses, true); 
    torch::Tensor H_base = Rotation(pmc_b, pmc_mu, &base);
    torch::Tensor Pi2 = operators_::Pi_2(&base); 
    torch::Tensor Derivative = operators_::Rz(&Pi2); 
    Derivative = Derivative.matmul(Shape(&Pi2, {1, 1, 0})); 

    // Invert the input sigma matrix 
    torch::Tensor _sigma = Sigma(met_xy, sigma); 
   
    // ------- Convert missing energy matrix into tensor ----- //
    torch::Tensor _met_xy = _met(met_xy); 

    // ------- create ellipse ------- //
    torch::Tensor dNu, X_, M_; 
    dNu = _met_xy - H_base; 
    X_ = torch::matmul(torch::transpose(dNu, 1, 2), _sigma); 
    X_ = torch::matmul(X_, dNu).view({-1, 3, 3}); 
    M_ = X_.matmul(Derivative); 
    M_ = M_ + torch::transpose(M_, 1, 2);  
    
    std::map<std::string, torch::Tensor> output; 
    output["M"] = M_;
    output["H"] = H_base; 
    output["X"] = X_; 
    return output; 
}

std::map<std::string, torch::Tensor> nusol_::Nu(
        torch::Tensor* pmc_b , torch::Tensor* pmc_mu, torch::Tensor* met_xy, 
        torch::Tensor* masses, torch::Tensor* sigma, double null)
{
    torch::Tensor v, chi2; 
    std::map<std::string, torch::Tensor> _Nu, output; 
    std::tuple<torch::Tensor, torch::Tensor> sols; 
    const unsigned int dim_i = met_xy -> size(0); 

    _Nu = _xNu(pmc_b, pmc_mu, met_xy, masses, sigma); 
    torch::Tensor M = _Nu["M"]; 
    torch::Tensor H = _Nu["H"]; 
    torch::Tensor X = _Nu["X"]; 

    torch::Tensor circl = Shape(met_xy, {1, 1, -1}); 
    sols = nusol_::Intersection(&M, &circl, null); 
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

    output["nu1"] = v; 
    output["distance"] = chi2;
    return output; 
}

std::map<std::string, torch::Tensor> nusol_::NuNu(
        torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, 
        torch::Tensor* pmc_l1, torch::Tensor* pmc_l2, 
        torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2)
{
    torch::Tensor H1, H2, H1_perp, H2_perp, N1, N2, S, n_, none;
    torch::Tensor zero, _v, __v, _d; 
    const torch::TensorOptions op = MakeOp(met_xy); 

    std::map<std::string, torch::Tensor> output; 
    std::tuple<torch::Tensor, torch::Tensor> sols; 

    // ---------------- Prepare all needed matrices ----------------- //
    if (!m2){m2 = m1;}
    H1 = nusol_::BaseMatrix(pmc_b1, pmc_l1, m1, true); 
    H2 = nusol_::BaseMatrix(pmc_b2, pmc_l2, m2, true);

    // --- protection against non-invertible matrices --- //
    none =  (torch::det(H1) == 0);  
    none += (torch::det(H2) == 0);
    none = none == 0; 
    output["NoSols"] = none; 
    const unsigned int dim_i = met_xy -> index({none}).size(0); 

    if (dim_i == 0){ 
        output["n_"]   = torch::zeros({1, 3}, op); 
        output["nu_1"] = torch::zeros({1, 3}, op); 
        output["nu_2"] = torch::zeros({1, 3}, op);        
        output["H_perp_1"] = H1; 
        output["H_perp_2"] = H2;
        return output; 
    }
   
    torch::Tensor xb1_ = pmc_b1 -> index({none}); 
    torch::Tensor xl1_ = pmc_l1 -> index({none}); 
    torch::Tensor h1_  = H1.index({none}); 
    H1 = Rotation(&xb1_, &xl1_, &h1_); 

    torch::Tensor xb2_ = pmc_b2 -> index({none}); 
    torch::Tensor xl2_ = pmc_l2 -> index({none}); 
    torch::Tensor h2_  = H2.index({none}); 
    H2 = Rotation(&xb2_, &xl2_, &h2_); 
 
    H1_perp = _H_perp(&H1); 
    H2_perp = _H_perp(&H2); 
      
    N1 = _N(&H1_perp); 
    N2 = _N(&H2_perp); 

    S = met_xy -> index({none}); 
    S = _met(&S) - Shape(&S, {1, 1, -1}); 
    n_ = torch::matmul(torch::matmul(S.transpose(1, 2), N2), S); 
    
    // ---------------- Start Algorithm ----------------- //
    sols = nusol_::Intersection(&N1, &n_, null); 
    torch::Tensor d = std::get<1>(sols); 
    torch::Tensor v = std::get<0>(sols).view({dim_i, -1, 1, 3}); 
    torch::Tensor v_ = torch::sum(S.view({dim_i, 1, 3, 3}) * v, -1).view({dim_i, -1, 1, 3}); 

    torch::Tensor K1 = torch::matmul(H1, torch::inverse(H1_perp)).view({dim_i, 1, 3, 3});
    torch::Tensor K2 = torch::matmul(H2, torch::inverse(H2_perp)).view({dim_i, 1, 3, 3});

    v  = (K1 * v).sum(-1);
    v_ = (K2 * v_).sum(-1);

    unsigned int dim_j  = v.size(1); 
    unsigned int dim_k  = v.size(2); 
    unsigned int dim_i_ = none.size(0); 
    
    zero = torch::zeros({dim_i_, dim_j, dim_k}, op); 
    _d = torch::zeros({dim_i_, dim_j}, op); 
    _d.index_put_({none}, d); 
    
    _v = zero.clone(); 
    _v.index_put_({none}, v); 

    __v = zero.clone(); 
    __v.index_put_({none}, v_); 
    
    output["nu_1"] = _v;  
    output["nu_2"] = __v; 
    output["distance"] = _d; 

    output["n_"] = n_; 
    output["H_perp_1"] = H1_perp; 
    output["H_perp_2"] = H2_perp; 
    return output; 
}

std::map<std::string, torch::Tensor> nusol_::NuNu(
            torch::Tensor* pmc_b1, torch::Tensor* pmc_b2, torch::Tensor* pmc_mu1, torch::Tensor* pmc_mu2,
            torch::Tensor* met_xy, double null, torch::Tensor* m1, torch::Tensor* m2,
            const double step, const double tolerance, const unsigned int timeout)
{
    return {}; 
}

std::map<std::string, torch::Tensor> nusol_::combinatorial(
               torch::Tensor* edge_index, torch::Tensor* batch , torch::Tensor* pmc, 
               torch::Tensor* pid       , torch::Tensor* met_xy, 
               double mT , double mW, double null, double perturb, 
               long steps, bool gev
){
    return {}; 
}

std::map<std::string, torch::Tensor> nusol_::BaseMatrix(torch::Tensor* pmc_b, torch::Tensor* pmc_mu, torch::Tensor* masses){
    return {}; 
}
