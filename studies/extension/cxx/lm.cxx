#include "lm.h"

LevenbergMarquardt::LevenbergMarquardt(nunu* nux, double** mx, double l0, double lMx, double lMn, long mxiter){
    this -> lambda0 = l0;   this -> lambdaMx = lMx; 
    this -> lambdaMn = lMn; this -> mxiter = mxiter; 
    this -> nux_ = nux; 
    this -> lambda_c = l0; 
    this -> _params = mx;
}

LevenbergMarquardt::~LevenbergMarquardt(){
    if (this -> _params){clear(this -> _params, 1, 4);}
}

double** LevenbergMarquardt::jacobian(double** param){
//    const int l = 4; 
//    const int lx = 6; 
//    double lambda = 1e-3;
//    double F[l], F_new[l];
//    double J[l][lx];  
//    double h = 1e-6; 
//    double tol = 1e-4; 
//    for (int ix(0); ix < this -> mxiter; ++ix) {
//        //compute_F(par, F);
//        double S = 0;
//        for (int i(0); i < l; ++i){S += F[i]*F[i];}
//        for (int j = 0; j < lx; j++) {
//            //Parameters par_plus = *par;
//            //par_plus.p[j] += h;
//            //compute_F(&par_plus, F_new);
//            for (int i(0); i < l; ++i){J[i][j] = (F_new[i] - F[i]) / h;}
//        }
//        
//        // Compute gradient and Hessian approximation
//        double JtJ[lx][lx] = {{0}};
//        double JtF[lx] = {0};
//        
//        for (int i(0); i < lx; ++i){
//            for (int k(0); k < l; ++k){
//                for (int j(0); j < lx; ++j){JtJ[i][j] += J[k][i] * J[k][j];}
//                JtF[i] += J[k][i] * F[k];
//            }
//        }
//        
//        double JtJ_damped[lx][lx];
//        for (int i(0); i < lx; ++i) {
//            for (int j(0); j < lx; ++j) {
//                JtJ_damped[i][j] = JtJ[i][j] + (i == j ? lambda * JtJ[i][i] : 0);
//            }
//        }
//        
//        double dp[lx];
//        for (int i(0); i < lx; ++i){dp[i] = -JtF[i] / (JtJ_damped[i][i] + 1e-10);}
//        
//        //Parameters par_new = *par;
//        //for (int i(0); i < lx; i++){par_new.p[i] += delta.p[i];}
//        
//        //compute_F(&par_new, F_new);
//        double S_new = 0;
//        for (int i(0); i < l; ++i){S_new += F_new[i]*F_new[i];}
//        if (S_new < S) {
//            //*par = par_new;
//            lambda /= 10;
//        } 
//        else {lambda *= 10;}
//        if (sqrt(S_new) < tol){break;}
//    }
//
    return nullptr;
}


double LevenbergMarquardt::step(){
    int lx = 0; int jx = 0; int ix = 0; 
    double** lss = this -> nux_ -> loss(&lx); 
    double   lxc = this -> loss(lss, lx); 
    double** J   = this -> nux_ -> jacobian(&ix, &jx); 
    double** jT  = T(J, ix, jx);

    double** jTj = dot(jT,   J, false, jx, ix, ix, jx);
    double** jTr = dot(jT, lss, false, jx, ix, lx, 1);
    jTr = scale(jTr, jx, 1, -1);

    clear(lss, lx, 1); 
    clear(J , ix, jx); 
    clear(jT, jx, ix); 

    double** dgx = diag(jTj, jx);
    double** dxl = arith(jTj, dgx, this -> lambda_c, jx, jx);
    double** delta  = nullptr; //dot(inv4(dxl), jTr, true, jx, jx, jx, 1);
    clear(dgx, jx, jx); clear(dxl, jx, jx);
    clear(jTj, jx, jx); clear(jTr, jx, 1); 

    double** deltaT = T(delta, jx, 1); 
    double** npr = arith(deltaT, this -> _params, 1.0, 1, jx);
    this -> nux_ -> update(npr); 
    clear(delta, jx, 1); 
    clear(deltaT, 1, jx); 

    lss = this -> nux_ -> loss(&lx); 
    double lxn = this -> loss(lss, lx); 
    clear(lss, lx, 1); 
    
    double lxx = 0;  
    if (lxc > lxn){
        this -> lambda_c *= 1.0/10.0; 
        clear(this -> _params, 1, jx); 
        this -> _params = npr; 
        lxx = lxn; 
    }
    else {
        this -> lambda_c *= 1.01;
        this -> nux_ -> update(this -> _params); 
        clear(npr, 1, jx); 
        lxx = lxc; 
    }
    this -> lambda_c = (this -> lambda_c > this -> lambdaMx) ? this -> lambdaMx : this -> lambda_c; 
    this -> lambda_c = (this -> lambda_c < this -> lambdaMn) ? this -> lambdaMn : this -> lambda_c; 
    return lxx; 
}

double LevenbergMarquardt::loss(double** param, int lx){
    double mx = 0; 
    for (int x(0); x < lx; ++x){mx += pow(param[x][0], 2);} 
    return mx; 
}

void LevenbergMarquardt::optimize(){
    for (int x(0); x < this -> mxiter; ++x){
        double lss = this -> step();
        std::cout << "-> " << lss*0.001 << " λ: " << this -> lambda_c << std::endl;
    }
    this -> nux_ -> update(nullptr); 
}



