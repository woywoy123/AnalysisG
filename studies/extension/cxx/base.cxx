#include "base.h"

nusol::nusol(particle* b, particle* l, double mW, double mT, bool delP){
    this -> b = b; 
    this -> l = l;
    this -> mw = mW; 
    this -> mt = mT; 
    this -> _s = sintheta(this -> b, this -> l); 
    this -> _c = costheta(this -> b, this -> l); 
    this -> delp = delP; 
}

nusol::~nusol(){
    if (this -> delp){delete this -> b; delete this -> l;}
    if (this -> h      ){delete this -> h;}
    if (this -> r_t    ){delete this -> r_t;}
    if (this -> dw_H   ){delete this -> dw_H;}
    if (this -> dt_H   ){delete this -> dt_H;}
    if (this -> h_perp ){delete this -> h_perp;}
    if (this -> h_tilde){delete this -> h_tilde;}
    if (this -> n_matrx){delete this -> n_matrx;}
    if (this -> k      ){delete this -> k;}
}

void nusol::flush(){
    if (this -> h      ){delete this -> h;}
    if (this -> dw_H   ){delete this -> dw_H;}
    if (this -> dt_H   ){delete this -> dt_H;}
    if (this -> h_perp ){delete this -> h_perp;}
    if (this -> h_tilde){delete this -> h_tilde;}
    if (this -> n_matrx){delete this -> n_matrx;}
    if (this -> k      ){delete this -> k;}
    this -> h       = nullptr; 
    this -> dw_H    = nullptr; 
    this -> dt_H    = nullptr; 
    this -> h_perp  = nullptr; 
    this -> h_tilde = nullptr; 
    this -> n_matrx = nullptr; 
    this -> k       = nullptr; 
}

void nusol::update(double mt_, double mw_){
    if (mw_){this -> mw = mw_;} 
    if (mt_){this -> mt = mt_;}
    this -> flush(); 
}


void nusol::misc(){
    std::cout << "--------- Neutrino ------------" << std::endl;
//    std::cout << "Sx:      " << this -> Sx() << std::endl;
//    std::cout << "dSx_dmW: " << this -> dSx_dmW() << std::endl;
//    std::cout << "Sy:      " << this -> Sy() << std::endl;         
//    std::cout << "dSy_dmW: " << this -> dSy_dmW() << std::endl;  
//    std::cout << "dSy_dmT: " << this -> dSy_dmT() << std::endl;  
//    std::cout << "w:       " << this -> w() << std::endl;        
//    std::cout << "w2:      " << this -> w2() << std::endl;       
//    std::cout << "om2:     " << this -> om2() << std::endl;      
//    std::cout << "Z:       " << this -> Z() << std::endl;        
//    std::cout << "Z2:      " << this -> Z2() << std::endl;       
//    std::cout << "dZ_dmT:  " << this -> dZ_dmT() << std::endl;   
//    std::cout << "dZ_dmW:  " << this -> dZ_dmW() << std::endl;   
//    std::cout << "x1:      " << this -> x1() << std::endl;       
//    std::cout << "dx1_dmW: " << this -> dx1_dmW() << std::endl;  
//    std::cout << "dx1_dmT: " << this -> dx1_dmT() << std::endl;  
//    std::cout << "y1:      " << this -> y1() << std::endl;       
//    std::cout << "dy1_dmW: " << this -> dy1_dmW() << std::endl;  
//    std::cout << "dy1_dmT: " << this -> dy1_dmT() << std::endl;  
//
    double mw1, mw2; 
    this -> r_mW(&mw1, &mw2); 
    std::cout << "mW root: " << mw1 << " " << mw2 << std::endl; 

    this -> get_mw(&mw1, &mw2); 
    std::cout << "mW derivative = 0: " << mw1 << " " << mw2 << std::endl; 

    this -> get_mt(&mw1, &mw2); 
    std::cout << "mT derivative = 0: " << mw1 << " " << mw2 << std::endl; 

    double A, B, C; 
    this -> Z2_coeff(&A, &B, &C); 
    std::cout << "Z2 A: " << A << " B: " << B << " C: " << C << std::endl; 

    std::cout << "N" << std::endl;
    this -> N() -> print(); 

    std::cout << "H" << std::endl; 
    this -> H() -> print(); 

    std::cout << "H_perp" << std::endl; 
    this -> H_perp() -> print();    

    std::cout << "H_tilde" << std::endl; 
    this -> H_tilde() -> print(); 

    std::cout << "dH_dmW" << std::endl; 
    this -> dH_dmW() -> print();

    std::cout << "dH_dmT" << std::endl; 
    this -> dH_dmT() -> print(); 

    std::cout << "K" << std::endl; 
    this -> K() -> print(); 

    std::cout << "RT" << std::endl; 
    this -> R_T() -> print();
}

mtx* nusol::R_T(){  
    if (this -> r_t){return this -> r_t;}
    double phi_mu   = this -> l -> phi(); 
    double theta_mu = this -> l -> theta();
    mtx rz  = mtx(3, 3).Rz(-phi_mu); 
    mtx rzt = rz.T(); 
    mtx ry  = mtx(3, 3).Ry(0.5*M_PI - theta_mu); 
    mtx ryt = ry.T(); 

    mtx bv = mtx(3, 1); 
    bv._m[0][0] = this -> b -> px;
    bv._m[1][0] = this -> b -> py;
    bv._m[2][0] = this -> b -> pz; 

    mtx vz = rz.dot(bv); 
    mtx vy = ry.dot(vz); 
    double  psi = -atan2(vy._m[2][0], vy._m[1][0]); 

    mtx rx  = mtx(3, 3).Rx(psi); 
    mtx rxt = rzt.dot(ryt.dot(rx.T()));
    this -> r_t = new mtx(rxt); 
    return this -> r_t; 
}

double nusol::Sx(){
    double p_mu = this -> l -> p(); 
    double b_mu = this -> l -> beta(); 
    double m2_mu = this -> l -> m2(); 

    double x0 = (m2_mu - pow(this -> mw, 2))/(2*this -> l -> e); 
    double sx = (x0 * b_mu - p_mu * (1 - pow(b_mu, 2))) / pow(b_mu, 2); 
    return sx;
}

double nusol::dSx_dmW(){
    return -this -> mw / (this -> l -> e * this -> l -> beta()); 
}


double nusol::Sy(){
    double x0p = -(pow(this -> mt, 2) - pow(this -> mw, 2) - this -> b -> m2())/ (2 * this -> b -> e); 
    return (x0p / this -> b -> beta() - this -> _c * this -> Sx()) / this -> _s; 
}

double nusol::dSy_dmW(){
    double v = this -> mw / (this -> b -> e * this -> b -> beta()); 
    return (v - this -> _c * this -> dSx_dmW())/this -> _s; 
}

double nusol::dSy_dmT(){
    return -this -> mt / (this -> b -> e * this -> b -> beta() * this -> _s); 
}

double nusol::w(){
    return (this -> l -> beta() / this -> b -> beta() - this -> _c)/ this -> _s;
}

double nusol::w2(){
    return pow(this -> w(), 2);
}

double nusol::om2(){
    return this -> w2() + 1 - this -> l -> beta2();
}

double nusol::x1(){
    double sx = this -> Sx();
    double sy = this -> Sy(); 
    return sx - (sx + this -> w() * sy)/this -> om2();
}

double nusol::dx1_dmW(){
    double v = this -> dSx_dmW() * (1 - 1 / this -> om2()); 
    return v - (this -> w()/this -> om2())*this -> dSy_dmW(); 
}

double nusol::dx1_dmT(){
    return -(this -> w()/this -> om2()) * this -> dSy_dmT();
}

double nusol::y1(){
    double sy = this -> Sy(); 
    double w_ = this -> w(); 
    return sy - (this -> Sx() + w_ * sy)*w_/this -> om2();
}

double nusol::dy1_dmW(){
    double v = this -> dSy_dmW()*(1 - this -> w2()/this -> om2()); 
    return v - (this -> w()/this -> om2())*this -> dSx_dmW();
}

double nusol::dy1_dmT(){
    return this -> dSy_dmT() * (1 - this -> w2()/this -> om2());
}


void nusol::Z2_coeff(double* A, double* B, double* C){
    double m2_m   = this -> l -> m2(); 
    double beta_m = this -> l -> beta();
    double beta_b = this -> b -> beta(); 
    double D1     = -(m2_m + pow(this -> mt, 2) - this -> b -> m2())/(2 * this -> b -> e * this -> _s * beta_b); 
    double D2     = -(this -> l -> e * beta_m / (this -> b -> e * beta_b) + this -> _c)/ this -> _s;  
    double w      = this -> w();
    double o2     = this -> om2(); 
    double P      = 1 - (1+w*D2)/o2; 
    *A = pow(P, 2) * o2 - pow(D2 - w, 2) + pow(beta_m, 2); 
    *B = 2*(-w*P*D1 - D1*(D2 - w) + beta_m * this -> l -> e); 
    *C = pow((-w * D1 / o2), 2)*o2 - pow(D1, 2) + m2_m; 
}

double nusol::Z2(){
    double A, B, C; 
    double sx = this -> Sx(); 
    this -> Z2_coeff(&A, &B, &C);
    return A*pow(sx, 2) + B*sx + C; 
}

double nusol::Z(){
    double z  = this -> Z2(); 
    z = (z > 0) ? pow(z, 0.5) : -pow(fabs(z), 0.5); 
    z = (z) ? z : 1;
    return z; 
}


double nusol::dZ_dmW(){
    double A, B, C; 
    double z = this -> Z(); 
    this -> Z2_coeff(&A, &B, &C); 
    return (2 * A * this -> Sx() + B) * this -> dSx_dmW() /(2 * (z + (z == 0))); 
}

void nusol::r_mW(double* mw1, double* mw2){
    auto sx_w = [this](double sx) -> double{
        sx *= this -> l -> beta2(); 
        sx += this -> l -> p() * (1 - this -> l -> beta2()); 
        sx /= this -> l -> beta();
        sx *= 2*this -> l -> e; 
        return -(sx - this -> l -> m2()); 
    }; 

    double A, B, C; 
    this -> Z2_coeff(&A, &B, &C); 
    double dsc = B*B - 4*A*C;  

    if (dsc < 0){return;}
    double _mw1 = sx_w((-B + pow(dsc, 0.5))/(2*A)); 
    double _mw2 = sx_w((-B - pow(dsc, 0.5))/(2*A)); 
    *mw1 = (_mw1 < 0) ? 0 : pow(_mw1, 0.5); 
    *mw2 = (_mw2 < 0) ? 0 : pow(_mw2, 0.5); 

}

double nusol::dZ_dmT(){
    double sx = this -> Sx(); 
    double D1 = -(this -> l -> m2() + pow(this -> mt, 2) - this -> b -> m2()); 
    D1 = D1/(2 * this -> b -> e * this -> b -> beta() * this -> _s); 
    double D2 = (this -> l -> e * this -> l -> beta());
    D2 =   D2 / (this -> b -> e * this -> b -> beta());
    D2 = -(D2 + this -> _c)/this -> _s; 

    double dD1_dmT = this -> dSy_dmT();     
    double P = 1 - (1 + this -> w()*D2)/this -> om2(); 

    double dB = -2*dD1_dmT*(this -> w() * P + (D2 - this -> w())); 
    double dC = -2*dD1_dmT*(1 - this -> w2()/this -> om2())*D1; 
    double z = this -> Z(); 
    return (dB*sx + dC)/(2 * (z + (z == 0))); 
}

mtx* nusol::H(){
    if (this -> h){return this -> h;}
    mtx h = this -> H_tilde() -> copy(); 
    mtx m = this -> R_T() -> dot(h); 
    this -> h = new mtx(&m); 
    return this -> h; 
}

mtx* nusol::H_tilde(){
    if (this -> h_tilde){return this -> h_tilde;}
    double z = this -> Z(); 
    this -> h_tilde = new mtx(3, 3); 
    this -> h_tilde -> _m[0][0] = z / pow(this -> om2(), 0.5); 
    this -> h_tilde -> _m[1][0] = this -> w() * z / pow(this -> om2(), 0.5); 
    this -> h_tilde -> _m[2][1] = z; 
    this -> h_tilde -> _m[0][2] = this -> x1() - this -> l -> p(); 
    this -> h_tilde -> _m[1][2] = this -> y1(); 
    return this -> h_tilde; 
}

mtx* nusol::H_perp(){
    if (this -> h_perp){return this -> h_perp;}
    mtx* h = this -> H(); 
    this -> h_perp = new mtx(3, 3);
    this -> h_perp -> copy(h, 0, 3); 
    this -> h_perp -> copy(h, 1, 3); 
    this -> h_perp -> _m[2][2] = 1.0; 
    return this -> h_perp; 
}

mtx* nusol::N(){
    if (this -> n_matrx){return this -> n_matrx;}
    mtx inv_n = this -> H_perp() -> inv();
    mtx inv_t = inv_n.T(); 
    mtx* circl = unit(); 
    this -> n_matrx = new mtx(inv_t.dot(*circl).dot(inv_n)); 
    delete circl;
    return this -> n_matrx; 
}


mtx* nusol::dH_dmW(){
    if (this -> dw_H){return this -> dw_H;}
    double dmW = this -> dZ_dmW(); 
    double omg = pow(this -> om2(), 0.5); 
    mtx dw_H = mtx(3, 3); 
    dw_H._m[0][0] = dmW/omg; 
    dw_H._m[0][2] = this -> dx1_dmW(); 
    dw_H._m[1][0] = this -> w()*dmW/omg; 
    dw_H._m[1][2] = this -> dy1_dmW();
    dw_H._m[2][1] = dmW; 
    this -> dw_H = new mtx(this -> R_T() -> dot(dw_H)); 
    return this -> dw_H; 
}

mtx* nusol::dH_dmT(){
    if (this -> dt_H){return this -> dt_H;}
    double dmT = this -> dZ_dmT(); 
    double omg = pow(this -> om2(), 0.5); 
    mtx mx = mtx(3, 3); 
    mx._m[0][0] = dmT/omg; 
    mx._m[0][2] = this -> dx1_dmT(); 
    mx._m[1][0] = this -> w()*dmT/omg; 
    mx._m[1][2] = this -> dy1_dmT();
    mx._m[2][1] = dmT; 
    this -> dt_H = new mtx(this -> R_T() -> dot(mx));
    return this -> dt_H; 
}

mtx* nusol::K(){
    if (this -> k){return this -> k;}
    mtx kx = this -> H_perp() -> inv(); 
    this -> k = new mtx(this -> H() -> dot(this -> H_perp() -> inv())); 
    return this -> k; 
}


void nusol::get_mw(double* v_crit, double* v_infl){
    double w_ = this -> w(); 
    double o2 = this -> om2(); 

    double e0 = this -> l -> m2()/(2.0 * this -> l -> e); 
    double e1 = -1.0 / (2.0 * this -> l -> e); 
    double p0 = (this -> b -> m2() - pow(this -> mt, 2)) / (2.0 * this -> b -> e); 
    double p1 =  1.0 / (2.0 * this -> b -> e * this -> b -> beta()); 

    double sx  =  e0 - this -> l -> m2()/this -> l -> e; 
    double sy0 = (p0/this -> b -> beta() - this -> _c * sx)/this -> _s; 

    double sy1 = (p1 - this -> _c * e1)/this -> _s; 
    double x0  = sx * (1.0 - 1.0/o2) - (w_ * sy0)/o2;
    double x1  = e1 * (1.0 - 1.0/o2) - (w_ * sy1)/o2;
    double d0  = sy0 - w_ * sx; 
    double d1  = sy1 - w_ * e1; 

    double a = o2 * pow(x1, 2) - pow(d1, 2) + pow(e1, 2); 
    double b = 2.0 * (o2 * x0 * x1 - d0 * d1) + 2 * e0 * e1 - this -> l -> beta2(); 
    double vc = -b / (2.0 * (a + (a == 0))); 
    double vi = -b / (6.0 * (a + (a == 0))); 
    *v_crit = (vc > 0) ? pow(vc, 0.5) : this -> mw;
    *v_infl = (vi > 0) ? pow(vi, 0.5) : this -> mw;
}

void nusol::get_mt(double* mt1, double* mt2){
    double sx = this -> Sx(); 
    double w_ = this -> w(); 
    double s  = this -> _s; 
    double c  = this -> _c; 
    double o2 = this -> om2(); 
    double mw2 = this -> mw * this -> mw; 
    double pb  = this -> b -> e * this -> b -> beta(); 

    double ms = mw2 + this -> b -> m2();
    double mk = - 2*this -> b -> p() * sx * (c + (1.0/(1 - this -> l -> beta2())) * w_ * s); 
    double mT2 = (1.0/3.0)*(ms + mk); 


    double a = -1.0 / (2.0 * pb * s); 
    double b = ((mw2 + this -> b -> m2())/(2.0 * pb) - c*sx)/s; 
    double a1 = -(w_ * a)/o2;
    double b1 = (sx * (o2 - 1) - w_ * b)/o2;
    double b2 = b - w_*sx;
    double A = o2 * pow(a1, 2) - pow(a, 2);
    double B = 2 * o2 * a1 * b1 - 2.0*a*b2; 
    double dzdt = -B / (2.0*A); 
    
    *mt1 = (dzdt > 0) ? pow(dzdt, 0.5) : -pow(fabs(dzdt), 0.5);
    *mt2 = (mT2 > 0) ?  pow(mT2, 0.5)  : -pow(fabs(mT2), 0.5);
}


