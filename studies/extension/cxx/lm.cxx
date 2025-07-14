#include "lm.h"

LevenbergMarquardt::LevenbergMarquardt(nunu* nux, double** mx, double l0, double lMx, double lMn, long mxiter){
    this -> lambda0 = l0;   this -> lambdaMx = lMx; 
    this -> lambdaMn = lMn; this -> mxiter = mxiter; 
    this -> nux_ = nux; 
    this -> lambda_c = l0; 
    this -> _params = mx;
}

LevenbergMarquardt::~LevenbergMarquardt(){}

double** LevenbergMarquardt::jacobian(double** param){


    return nullptr; 
}

double LevenbergMarquardt::step(){
    int lx = 0; int jx = 0; int ix = 0; 
    double** lss = this -> nux_ -> loss(&lx); 
    double   lxc = this -> loss(lss[0], lx); 
    double** J   = this -> nux_ -> jacobian(&ix, &jx); 
    double** jT  = T(J, ix, jx);
    std::cout << "....." << std::endl;

    double** jTj = dot(jT, J, false, jx, ix, ix, jx);
    double** jTr = dot(jT, lss, true, jx, ix, lx, 1);
    jTr = scale(jTr, jx, 1, -1);
    clear(lss, lx, 1); 

    double** dgx = diag(jTj, jx);
    double** dxl = arith(jTj, dgx, this -> lambda_c, jx, jx);
    double** delta  = dot(inv4(dxl), jTr, true, jx, jx, jx, 1);
    clear(dgx, jx, jx); clear(dxl, jx, jx);

    double** deltaT = T(delta, jx, 1); 
    double** npr = arith(deltaT, this -> _params, 1.0, 1, jx);
    double** red = dot(deltaT, jTr, false, 1, jx, jx, 1);  
    double** rdd = dot(deltaT, jTj, true, 1, jx, jx, jx);
    clear(jTj, jx, jx); clear(jTr, jx, 1); 

    rdd = dot(rdd, delta, true, 1, jx, jx, 1); 
    double pred_red = -red[0][0] - 0.5*rdd[0][0]; 
    clear(red, 1, 1); clear(rdd, 1, 1); 
    clear(delta, jx, 1); 

    print_(npr, 1, jx); 
    this -> nux_ -> flush(); 
    this -> nux_ -> update(npr); 

    lss = this -> nux_ -> loss(&lx); 
    double lxn = this -> loss(lss[0], lx); 
    clear(lss, 1, lx); 

    double axn = lxc - lxn; 
    double rho = (pred_red > 0) ? axn / (pred_red + 1e-16) : 0; 
 
    double lxx = 0;  
    if (rho > 0){
        clear(this -> _params, 1, jx); 
        rho = pow(2*rho - 1, 3); 
        this -> _params = npr; 
        this -> lambda_c *= (1.0/3.0 > rho) ? 1.0/3.0 : rho; 
        lxx = lxn; 
    }
    else {
        this -> lambda_c *= 2.0;
        lxx = lxc; 
        clear(npr, 1, jx); 
    }
    return lxx; 
}

double LevenbergMarquardt::loss(double* param, int lx){
    double mx = 0; 
    for (int x(0); x < lx; ++x){mx += pow(param[x], 2);} 
    return mx; 
}

void LevenbergMarquardt::optimize(){
    for (int x(0); x < this -> mxiter; ++x){
        double lss = this -> step();
        std::cout << lss << std::endl;
    }
}



