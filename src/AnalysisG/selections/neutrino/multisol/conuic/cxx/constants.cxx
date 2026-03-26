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

cline_t* build_clines(branches_t* br, delta_t* dt, double sign){
    long double dt_ = (sign < 0) ? dt -> dm : dt -> dp; 
    long double dtx = (sign < 0) ? dt -> dp : dt -> dm; 
    return new cline_t(br, dt_, sign, dtx);  
}

