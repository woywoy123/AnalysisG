#include "base.h"

nusol::nusol(particle* b, particle* l, double mW, double mT){
    this -> b = b; 
    this -> l = l;
    this -> mw = mW; 
    this -> mt = mT; 
    this -> _s = sintheta(this -> b, this -> l); 
    this -> _c = costheta(this -> b, this -> l); 
}

nusol::~nusol(){
    delete this -> b; delete this -> l;
    if (this -> h      ){clear(this -> h      , 3, 3);}
    if (this -> r_t    ){clear(this -> r_t    , 3, 3);}
    if (this -> dw_H   ){clear(this -> dw_H   , 3, 3);}
    if (this -> dt_H   ){clear(this -> dt_H   , 3, 3);}
    if (this -> h_perp ){clear(this -> h_perp , 3, 3);}
    if (this -> h_tilde){clear(this -> h_tilde, 3, 3);}
    if (this -> n_matrx){clear(this -> n_matrx, 3, 3);}
}


double** nusol::R_T(){  
    if (this -> r_t){return this -> r_t;}
    double phi_mu = this -> l -> phi(); 
    double theta_mu = this -> l -> theta();

    double** rz = matrix(3, 3);
    rz[0][0] =  std::cos(-phi_mu); 
    rz[0][1] = -std::sin(-phi_mu); 
    rz[1][0] =  std::sin(-phi_mu); 
    rz[1][1] =  std::cos(-phi_mu); 
    rz[2][2] = 1.0;

    double** ry = matrix(3, 3);
    ry[0][0] = std::cos(0.5*M_PI - theta_mu); 
    ry[0][2] = std::sin(0.5*M_PI - theta_mu); 
    ry[1][1] = 1.0; 
    ry[2][0] = -std::sin(0.5*M_PI - theta_mu); 
    ry[2][2] =  std::cos(0.5*M_PI - theta_mu);

    double** bv = matrix(3, 1); 
    bv[0][0] = this -> b -> px;
    bv[1][0] = this -> b -> py;
    bv[2][0] = this -> b -> pz; 

    double** vz = dot(rz, bv, 3, 3, 3, 1); 
    double** vy = dot(ry, vz, 3, 3, 3, 1);
    double  psi = -atan2(vy[2][0], vy[1][0]); 

    double** rx = matrix(3, 3);
    rx[0][0] = 1.0; 
    rx[1][1] =  std::cos(psi); 
    rx[1][2] = -std::sin(psi); 
    rx[2][1] =  std::sin(psi);
    rx[2][2] =  std::cos(psi); 

    double** rxt = T(rx, 3, 3); 
    double** ryt = T(ry, 3, 3); 
    double** rzt = T(rz, 3, 3); 
    double** rxy = dot(ryt, rxt, 3, 3, 3, 3); 
    double** ryz = dot(rzt, rxy, 3, 3, 3, 3);  
    clear(rz, 3, 3); 
    clear(ry, 3, 3);
    clear(bv, 3, 1); 
    clear(vz, 3, 1); 
    clear(vy, 3, 1);
    clear(rx, 3, 3); 
    clear(rxt, 3, 3); 
    clear(ryt, 3, 3);  
    clear(rzt, 3, 3); 
    clear(rxy, 3, 3);
    this -> r_t = ryz; 
    return ryz; 
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
    return pow(this -> w(), 2) + 1 - pow(this -> l -> beta(), 2);
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
    return (z > 0) ? pow(z, 0.5) : -pow(fabs(z), 0.5); 
}


double nusol::dZ_dmW(){
    double v = this -> dSx_dmW();
    double z = this -> Z(); 
    double A, B, C; 
    this -> Z2_coeff(&A, &B, &C); 
    return (2 * A * this -> Sx() + B) * this -> dSx_dmW() /(2 * (z + (z == 0))); 
}

double nusol::dZ_dmT(){
    double sx = this -> Sx(); 
    double D1 = -(this -> l -> m2() + pow(this -> mt, 2) - this -> b -> m2()); 
    D1 = D1/(2 * this -> b -> e * this -> _s * this -> b -> beta()); 

    double D2 = (this -> l -> e * this -> l -> beta());
    D2 = D2 / (this -> b -> e * this -> b -> beta());
    D2 = -(D2 + this -> _c)/this -> _s; 

    double dD1_dmT = this -> dSy_dmT();     
    double P = 1 - (1 + this -> w()*D2)/this -> om2(); 

    double dB = -2*dD1_dmT*(this -> w() * P + (D2 - this -> w())); 
    double dC = -2*dD1_dmT*(1 - this -> w2()/this -> om2())*D1; 
    double z = this -> Z(); 
    return (dB*sx + dC)/(2 * (z + (z == 0))); 
}



double** nusol::H(){
    if (this -> h){return this -> h;}
    this -> h = dot(this -> R_T(), this -> H_tilde(), 3, 3, 3, 3); 
    return this -> h; 
}

double** nusol::H_tilde(){
    if (this -> h_tilde){return this -> h_tilde;}
    double z2 = this -> Z2(); 
    double** _matrix = matrix(3, 3); 
    _matrix[0][0] = pow(z2 / this -> om2(), 0.5); 
    _matrix[1][0] = this -> w() * pow(z2 / this -> om2(), 0.5); 
    _matrix[2][1] = pow(z2, 0.5); 
    _matrix[0][2] = this -> x1() - this -> l -> p(); 
    _matrix[1][2] = this -> y1(); 
    this -> h_tilde = _matrix; 
    return this -> h_tilde; 
}

double** nusol::H_perp(){
    if (this -> h_perp){return this -> h_perp;}
    double** h = this -> H();
    double** m = matrix(3, 3); 
    for (int x(0); x < 3; ++x){m[0][x] = h[0][x];}
    for (int x(0); x < 3; ++x){m[1][x] = h[1][x];}
    m[2][2] = 1.0;  
    this -> h_perp = m;
    return this -> h_perp; 
}

double** nusol::N(){
    if (this -> n_matrx){return this -> n_matrx;}
    double** inv_n = inv(this -> H_perp()); 
    double** inv_T = T(inv_n, 3, 3); 
    double** circl = unit(); 
    double** tmp = dot(inv_T, circl, 3, 3, 3, 3); 
    this -> n_matrx = dot(tmp, inv_n, 3, 3, 3, 3); 
    clear(inv_T, 3, 3); 
    clear(inv_n, 3, 3);
    clear(circl, 3, 3); 
    clear(tmp  , 3, 3); 
    return this -> n_matrx; 
}


double** nusol::dH_dmW(){
    if (this -> dw_H){return this -> dw_H;}
    double dmW = this -> dZ_dmW(); 
    double omg = pow(this -> om2(), 0.5); 
    double** _matrix = matrix(3, 3);
    _matrix[0][0] = dmW/omg; 
    _matrix[0][2] = this -> dx1_dmW(); 
    _matrix[1][0] = this -> w()*dmW/omg; 
    _matrix[1][2] = this -> dy1_dmW();
    _matrix[2][1] = dmW; 
    this -> dw_H = dot(this -> R_T(), _matrix, 3, 3, 3, 3); 
    clear(_matrix, 3, 3); 
    return this -> dw_H; 
}

double** nusol::dH_dmT(){
    if (this -> dt_H){return this -> dt_H;}
    double dmT = this -> dZ_dmT(); 
    double omg = pow(this -> om2(), 0.5); 
    double** _matrix = matrix(3, 3);
    _matrix[0][0] = dmT/omg; 
    _matrix[0][2] = this -> dx1_dmT(); 
    _matrix[1][0] = this -> w()*dmT/omg; 
    _matrix[1][2] = this -> dy1_dmT();
    _matrix[2][1] = dmT; 
    this -> dt_H = dot(this -> R_T(), _matrix, 3, 3, 3, 3); 
    clear(_matrix, 3, 3); 
    return this -> dt_H; 
}

void nusol::get_mw(double* v_crit, double* v_infl){
    double w_ = this -> w(); 
    double o2 = this -> om2(); 

    double e0 = this -> l -> m2()/(2.0 * this -> l -> e); 
    double e1 = -1.0 / (2.0 * this -> l -> e); 
    double p1 =  1.0 / (2.0 * this -> b -> e * this -> b -> beta()); 
    double p0 = (this -> b -> m2() - pow(this -> mt, 2)) * p1; 

    double sx  =  e0 - this -> l -> m2()/this -> l -> e; 
    double sy0 = (p0 - this -> _c * sx)/this -> _s; 
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
    *v_infl = (vi > 0) ? pow(vc, 0.5) : this -> mw;
}

void nusol::get_mt(double* mt1, double* mt2){
    double sx = this -> Sx(); 
    double w_ = this -> w(); 
    double o2 = this -> om2(); 
    double pb = this -> b -> e * this -> b -> beta(); 
    double a = -1.0 / (2.0 * pb * this -> _s); 
    double b = ((pow(this -> mw, 2) + this -> b -> m2())/(2.0 * pb) - this -> _c*sx)/this -> _s; 
    double a1 = -(w_ * a)/o2;
    double b1 = (sx * (o2 - 1) - w_ * b)/o2;
    double b2 = b - w_*sx;
    double A = o2 * pow(a1, 2) - pow(a, 2);
    double B = 2 * o2 * a1 * b1 - 2.0*a*b2; 
    double dzdt = -B / (2.0*A); 
    double d2zd2t = -B / (6.0*A); 
    *mt1 =   (dzdt > 0) ?   pow(dzdt, 0.5) : -pow(fabs(dzdt  ), 0.5);
    *mt2 = (d2zd2t > 0) ? pow(d2zd2t, 0.5) : -pow(fabs(d2zd2t), 0.5);
}

