#include <conuic/constants.h>
#include <conuic/conuic.h>
#include <tools/tools.h>

branches_t* build_branches(kinematics_t* j, kinematics_t* l, int sign){
    branches_t* br = new branches_t(); 
    long double w = omega(j, l, sign);
    long double O = Omega(j, l, sign); 
    long double lb = l -> b;
    br -> w = w; br -> O = O;
    br -> cth = costh(j, l);  
    br -> sth = std::sqrt(1 - br -> cth * br -> cth); 
    br -> tth = br -> sth / br -> cth; 

    // ...... Z2 quadric coefficients ........ //
    br -> A = (lb * lb - w * w) / (O * O); 
    br -> B = 2 * w / (O * O);
    br -> C = w * w / (O * O) - 1; 
    br -> D = 2 * l -> p;
    br -> E = l -> m * l -> m; 

    // ......... Eigenvalues .............. //
    br -> l1 = -1L; 
    br -> l2 = lb * lb / ( O * O );
  
    // ......... Parameterizations for Sx and Sy ......... // 
    br -> tpsi = w; br -> cpsi = tn_cos(w); br -> spsi = tn_sin(w);
    br -> sx0 = -     l -> m * l -> m / l -> p; // m^2_mu / p_mu
    br -> sy0 = - w * l -> e / l -> b; // omega * E_mu / beta_mu 

    br -> bl = l -> b; 
    return br; 
}

void build_tilde(branches_t* br, kinematics_t* knl){
    matrix_t* CC = new matrix_t(3, 3); // cosh matrix
    CC -> at(0, 2) = - br -> cpsi * knl -> b / br -> O;
    CC -> at(1, 2) = - br -> spsi * knl -> b / br -> O; 
    br -> CC = CC; 

    matrix_t* SC = new matrix_t(3, 3); // sinh cos(kappa) matrix
    SC -> at(0, 2) = - br -> spsi;
    SC -> at(1, 2) =   br -> cpsi; 
    br -> SC = SC; 

    matrix_t* SS = new matrix_t(3, 3); // sinh sin(kappa) matrix
    SS -> at(0, 0) = 1 / br -> O;
    SS -> at(1, 0) = br -> tpsi / br -> O;
    SS -> at(2, 1) = 1; 
    br -> SS = SS; 
}

delta_t* build_deltas(branches_t* plus, branches_t* minus){
    delta_t* dt = new delta_t();
    dt -> dp = delta(plus, minus, +1);
    dt -> Gp = Gamma(plus, minus, +1);
    dt -> salp = tn_sin(dt -> dp); 
    dt -> calp = tn_cos(dt -> dp); 
    dt -> talp = dt -> dp; 
    
    dt -> dm = delta(plus, minus, -1); 
    dt -> Gm = Gamma(plus, minus, -1);
    dt -> salm = tn_sin(dt -> dm);  
    dt -> calm = tn_cos(dt -> dm);  
    dt -> talm = dt -> dm; 

    dt -> alp = std::atan(dt -> dp); 
    dt -> alm = std::atan(dt -> dm); 

    dt -> alpha_p = - (dt -> alp + dt -> alm) * 0.5L;
    dt -> alpha_m =   (dt -> alp - dt -> alm) * 0.5L;
 
    dt -> lp = lm_dt(dt, +1); 
    dt -> lm = lm_dt(dt, -1); 

    dt -> Glp = - dt -> Gm * dt -> Gp * dt -> lp; 
    dt -> Glm = - dt -> Gm * dt -> Gp * dt -> lm; 

    return dt; 
}


bool intersection(
        branches_t* pls, branches_t* msn, long double dt, 
        long double& Sx, long double& Sy, long double& m_nu
) {
    long double Pp = pls -> A * dt*dt + pls -> B * dt + pls -> C;
    long double Pm = msn -> A * dt*dt + msn -> B * dt + msn -> C;
    long double Qp = pls -> D * dt;
    long double Qm = msn -> D * dt;
    long double dP = Pp - Pm;
    if (std::fabs(dP) < 1e-12L){return false;}
    Sy = -(Qp - Qm) / dP;
    Sx = dt * Sy;

    long double R = -(Pp * Sy * Sy + Qp * Sy);
    long double mnu2 = std::sqrt(std::fabs(pls -> E - R));
    return true;
}

geometry_t build_ruling(branches_t* pls, branches_t* msn) {
    long double Sx, Sy, m_nu;
    long double dp = delta(pls, msn, +1);
    if (intersection(pls, msn, dp, Sx, Sy, m_nu)){}
    else {intersection(pls, msn, delta(pls, msn, -1), Sx, Sy, m_nu);}

    geometry_t geo;
    geo.Sx0 =  pls -> sx0; 
    geo.Sy0 = (pls -> sy0 + msn -> sy0) * 0.5L;
    long double cp = pls -> cpsi;
    long double sp = pls -> spsi;
    long double cm = msn -> cpsi;
    long double sm = msn -> spsi;

    // compute the skew angle using arctan2
    geo.l1 = cp - cm;
    geo.l2 = sp - sm;

    // distance between sheets.
    long double wp = pls -> w;
    long double wm = msn -> w;
    long double Op = pls -> O;
    long double Om = msn -> O;
    long double bl = pls -> bl; 
    geo.d = (std::fabs(m_nu) / bl) * std::fabs(wp - wm) * std::fabs(wp + wm) / (Op + Om);

    long double dp_ = delta(pls, msn, +1);
    long double dm_ = delta(pls, msn, -1);
    long double txa = Op * Om / std::fabs(dp_ * dm_ - wp * wm);
    if (dp_ + dm_ == 0){geo.tau = 0;}
    geo.tau = std::atanh((std::abs(txa) > 1) ? - 1 / txa : txa);

    geo.m_nu = std::fabs(m_nu);
    geo.alpha = std::atan2(sp + sm, cp + cm);
    return geo;
}

long double confocal(branches_t* plus, branches_t* minus, kinematics_t* mu) {
    long double wp = plus -> w, wm = minus -> w;
    long double Op = plus -> O, Om = minus -> O;
    long double bl2 = mu -> b * mu -> b;

    long double vp = 1.0L + (1.0L + wp * wp) / bl2 + (1.0L + wp * wp) * (1.0L + wp * wm) / (bl2 * Op * Om); 
    long double vm = 1.0L + (1.0L + wp * wp) / bl2 - (1.0L + wp * wp) * (1.0L + wp * wm) / (bl2 * Op * Om);
    // clamp to physical range
    if (vm > 1.0L) vm = 1.0L;
    if (vm < 0.0L) vm = 0.0L;
    std::cout << "-> " << vm << " " << vp << std::endl;
    return std::atanh(std::sqrt(vm)); 
}

void align(branches_t* plus, branches_t* minus, long double& dx, long double& dy) {
    long double a = plus->cpsi - minus->cpsi;
    long double b = plus->spsi - minus->spsi;

    // a vector along the line a*dx + b*dy = 0 is ( -b, a )
    dx = -b;
    dy =  a;
    // Normalise (optional)
    long double norm = std::sqrt(dx * dx + dy * dy);
    if (norm > 1e-12L) { dx /= norm; dy /= norm; }
}


special_t* build_special(branches_t* pls, branches_t* msn, delta_t* dt, kinematics_t* kln, conuic* db){
    special_t* spl = new special_t();
    spl -> nueq_dLpp = m_nueq_line(dt, pls, msn, kln, +1, false); 
    spl -> nueq_dLmm = m_nueq_line(dt, pls, msn, kln, -1, false); 
    spl -> nueq_dLpm = m_nueq_line(dt, pls, msn, kln, +1, true); 
    spl -> nueq_dLmp = m_nueq_line(dt, pls, msn, kln, -1, true); 

    std::cout << tools::to_string(spl -> nueq_dLpp, 12) << " " 
              << tools::to_string(spl -> nueq_dLmm, 12) << " "
              << tools::to_string(spl -> nueq_dLpm, 12) << " " 
              << tools::to_string(spl -> nueq_dLmp, 12) << std::endl;

    long double dfx, dfy; 
    
    geometry_t gx; 
    gx.tau = confocal(pls, msn, kln); 
    align(pls, msn, dfx, dfy); 

    matrix_t vx(3, 1);
    vx.at(0,0) = dfx; 
    vx.at(1,0) = dfy; 
    matrix_t vo = kln -> RT -> dot(vx).T(); 
    std::cout << gx.tau << std::endl;     


    geometry_t gxf = build_ruling(pls, msn); 
    std::cout << gxf.l1 << " " << gxf.l2 << std::endl;
    std::cout << gxf.Sx0 << " " << gxf.Sy0 << std::endl;

    long double kappa = std::acos(spl -> nueq_dLpp / std::tanh(gx.tau)); 
    long double m_nu = m_nuG(dt, pls, kln, gx.tau, kappa, +1, +1); 

    // Compute alignment rotation
    long double psi = std::atan2(vx.at(1, 0), vx.at(0, 0));
    matrix_t R_align = angular_t(psi).Rz(); // assuming z-axis fixed, rotation in xy-plane
    
    // Aligned neutrino momentum in F frame
    long double p_axis = (pls -> O / kln -> b) * m_nu;
    matrix_t p_nu_F_aligned(3, 1); 
    p_nu_F_aligned.at(0, 0) = p_axis;
   
    
    // Rotate back to original F frame
    matrix_t p_nu_F = R_align.T().dot(p_nu_F_aligned);
    
    // Rotate to lab frame using R_T (previously built)
    matrix_t p_nu_lab = kln -> RT -> dot(p_nu_F);
    p_nu_lab.print();

    std::cout << db -> x1(gx.tau, kappa, m_nu, +1, +1) 
              << " " << db -> y1(gx.tau, kappa, m_nu, +1, +1)
              << " " << db -> Z(gx.tau, kappa, m_nu, +1) << std::endl; 





    abort(); 
                



    std::cout << std::endl;  

    vx.print(); 
    vo.print(); 

//    std::cout << "a " << tools::to_string(gx.l1   , 12) << " " 
//              << "b " << tools::to_string(gx.l2   , 12) << " "
//
//              << "alpha " << tools::to_string(gx.alpha, 12) << " \n" 
//              << "Sx0 " << tools::to_string(gx.Sx0  , 12) << " "
//              << "Sy0 " << tools::to_string(gx.Sy0  , 12) << " " 
//              << "m_nu " << tools::to_string(gx.m_nu  , 12) << " " 
//
//              << "d "   << tools::to_string(gx.d    , 12) << " "
//              << "tau " << tools::to_string(gx.tau  , 12) << std::endl;
// 


    return spl; 
}


