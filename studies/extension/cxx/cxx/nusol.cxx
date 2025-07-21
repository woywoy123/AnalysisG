#include <templates/solvers.h>
#include <templates/nusol.h>
#include <templates/mtx.h>

double costheta(wrapper* p1, wrapper* p2){
    double pxx  = p1 -> px * p2 -> px + p1 -> py * p2 -> py + p1 -> pz * p2 -> pz; 
    return pxx / pow(p1 -> p2 * p2 -> p2, 0.5); 
}

double sintheta(wrapper* p1, wrapper* p2){
    return std::pow(1 - std::pow(costheta(p1, p2), 2), 0.5);
}

wrapper::wrapper(double px, double py, double pz, double e){
    this -> px = px; this -> py = py; this -> pz = pz; this -> e = e;
    this -> set(); 
}

void wrapper::set(){
    this -> p2 = std::pow(this -> px, 2) + std::pow(this -> py, 2) + std::pow(this -> pz, 2);
    this -> p  = std::pow(this -> p2, 0.5); 

    this -> m2 = std::pow(this -> e, 2) - this -> p2; 
    this -> m  = std::pow(this -> m, 0.5);

    this -> b   = this -> p / this -> e; 
    this -> b2  = std::pow(this -> b, 2); 
    this -> phi = std::atan2(this -> py, this -> px); 
    this -> theta = std::atan2(std::pow(pow(this -> px, 2) + std::pow(this -> py, 2), 0.5), this -> pz);
}

nusol::nusol(){}
nusol::nusol(wrapper* b_, wrapper* l_, double mW, double mT){
    this -> b = b_;  
    this -> l = l_; 
    this -> mw = mW; 
    this -> mt = mT; 
    this -> mw2 = mW*mW;
    this -> mt2 = mT*mT; 
    this -> _s = sintheta(this -> b, this -> l); 
    this -> _c = costheta(this -> b, this -> l); 
}

nusol::~nusol(){
    this -> flush();
    if (this -> b){delete this -> b;}
    if (this -> l){delete this -> l;}
    if (this -> r_t){delete this -> r_t;}
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

mtx* nusol::R_T(){  
    if (this -> r_t){return this -> r_t;}
    double phi_mu   = this -> l -> phi; 
    double theta_mu = this -> l -> theta;
    mtx* rz = Rz(-phi_mu); 
    mtx* ry = Ry(0.5*M_PI - theta_mu); 

    mtx bv = mtx(3, 1); 
    bv._m[0][0] = this -> b -> px;
    bv._m[1][0] = this -> b -> py;
    bv._m[2][0] = this -> b -> pz; 

    mtx vz = rz -> dot(bv); 
    mtx vy = ry -> dot(vz); 
    double  psi = -std::atan2(vy._m[2][0], vy._m[1][0]); 

    mtx* rx = Rx(psi);
    mtx rxt = rz -> T().dot(ry -> dot(rx -> T()));
    this -> r_t = new mtx(rxt); 
    delete rz; delete ry; delete rx; 
    return this -> r_t; 
}

double nusol::x0(){
    return (this -> l -> m2 - this -> mw2)/(2*this -> l -> e); 
}

double nusol::Sx(){
    return (this -> x0() * this -> l -> b - this -> l -> p * (1 - this -> l -> b2)) / this -> l -> b2; 
}

double nusol::Sy(){
    double x0p = -(this -> mt2 - this -> mw2 - this -> b -> m2)/ (2 * this -> b -> p); 
    return (x0p - this -> _c * this -> Sx()) / this -> _s; 
}

double nusol::w(){
    return (this -> l -> b / this -> b -> b - this -> _c)/ this -> _s;
}

double nusol::w2(){
    return pow(this -> w(), 2);
}

double nusol::om2(){
    return this -> w2() + 1 - this -> l -> b2;
}

double nusol::x1(){
    double sx = this -> Sx();
    double sy = this -> Sy(); 
    return sx - (sx + this -> w() * sy)/this -> om2();
}

double nusol::y1(){
    double sy = this -> Sy(); 
    double w_ = this -> w(); 
    return sy - (this -> Sx() + w_ * sy)*w_/this -> om2();
}

void nusol::Z2(double* A, double* B, double* C){
    double D1 = -(this -> l -> m2 + this -> mt2 - this -> b -> m2)/(2 * this -> b -> p * this -> _s); 
    double D2 = -(this -> l -> p / this -> b -> p + this -> _c)/ this -> _s;  
    double w_ = this -> w();
    double o2 = this -> om2(); 
    double P  = 1 - (1 + w_ * D2)/o2; 
    *A = pow(P, 2) * o2 - pow(D2 - w_, 2) + this -> l -> b2; 
    *B = 2*(-w_ * P * D1 - D1 * (D2 - w_) + this -> l -> p); 
    *C = pow((-w_ * D1 / o2), 2)*o2 - pow(D1, 2) + this -> l -> m2; 
}

double nusol::Z2(){
    double A, B, C; 
    double sx = this -> Sx(); 
    this -> Z2(&A, &B, &C);
    return A * sx * sx + B*sx + C; 
}

double nusol::Z(){
    double z  = this -> Z2(); 
    z = (z > 0) ? pow(z, 0.5) : -pow(fabs(z), 0.5); 
    z = (z) ? z : 1;
    return z; 
}

mtx* nusol::H(){
    if (this -> h){return this -> h;}
    mtx m = this -> R_T() -> dot(this -> H_tilde()); 
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
    this -> h_tilde -> _m[0][2] = this -> x1() - this -> l -> p; 
    this -> h_tilde -> _m[1][2] = this -> y1(); 
    return this -> h_tilde; 
}

mtx* nusol::H_perp(){
    if (this -> h_perp){return this -> h_perp;}
    mtx* h_ = this -> H(); 
    this -> h_perp = new mtx(3, 3);
    this -> h_perp -> copy(h_, 0, 3); 
    this -> h_perp -> copy(h_, 1, 3); 
    this -> h_perp -> _m[2][2] = 1.0; 
    return this -> h_perp; 
}

mtx* nusol::N(){
    if (this -> n_matrx){return this -> n_matrx;}
    mtx inv_n = this -> H_perp() -> inv();
    mtx inv_t = inv_n.T(); 
    mtx* circl = unit(); 
    this -> n_matrx = new mtx(inv_t.dot(circl).dot(inv_n)); 
    delete circl;
    return this -> n_matrx; 
}

mtx* nusol::K(){
    if (this -> k){return this -> k;}
    mtx kx = this -> H_perp() -> inv(); 
    this -> k = new mtx(this -> H() -> dot(this -> H_perp() -> inv())); 
    return this -> k; 
}

double nusol::dSx_dmW(){
    return -this -> mw / this -> l -> p; 
}

double nusol::dSy_dmW(){
    return ( this -> mw / this -> b -> p - this -> _c * this -> dSx_dmW() )/this -> _s; 
}

double nusol::dSy_dmT(){
    return -this -> mt / (this -> b -> p * this -> _s); 
}

double nusol::dx1_dmW(){
    double sx = this -> dSx_dmW(); 
    return sx - (1 / this -> om2()) * (sx + this -> w() * this -> dSy_dmW()); 
}

double nusol::dx1_dmT(){
    return -(this -> w()/this -> om2()) * this -> dSy_dmT();
}

double nusol::dy1_dmW(){
    double o2 = 1.0/this -> om2(); 
    return this -> dSy_dmW()*(1 - this -> w2() * o2) - ( this -> w() * o2 )*this -> dSx_dmW();
}

double nusol::dy1_dmT(){
    return this -> dSy_dmT() * (1 - this -> w2()/this -> om2());
}

double nusol::dZ_dmW(){
    double A, B, C; 
    double z = this -> Z(); 
    this -> Z2(&A, &B, &C); 
    return (2 * A * this -> Sx() + B) * this -> dSx_dmW() /(2 * (z + (z == 0))); 
}

double nusol::dZ_dmT(){
    double D1 = -(this -> l -> m2 + this -> mt2 - this -> b -> m2); 
    double D2 = this -> l -> p / this -> b -> p;
    D2 = -(D2 + this -> _c)/this -> _s; 

    double dD1_dmT = this -> dSy_dmT();     
    double P = 1 - (1 + this -> w()*D2)/this -> om2(); 

    double dB = dD1_dmT * (this -> w() * P + (D2 - this -> w())); 
    double dC = dD1_dmT * D1 * (1 - this -> w2()/this -> om2()) / (2 * this -> b -> p * this -> _s); 
    double z = this -> Z(); 
    return -(dB * this -> Sx() + dC)/(z + (z == 0)); 
}

mtx* nusol::dH_dmW(){
    if (this -> dw_H){return this -> dw_H;}
    double dmW = this -> dZ_dmW(); 
    double omg = pow(this -> om2(), 0.5); 
    mtx _dw_H = mtx(3, 3); 
    _dw_H._m[0][0] = dmW/omg; 
    _dw_H._m[0][2] = this -> dx1_dmW(); 
    _dw_H._m[1][0] = this -> w()*dmW/omg; 
    _dw_H._m[1][2] = this -> dy1_dmW();
    _dw_H._m[2][1] = dmW; 
    this -> dw_H = new mtx(this -> R_T() -> dot(_dw_H)); 
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

void nusol::r_mW(double* mw1_, double* mw2_){
    double p_l  = this -> l -> p;
    double p_b  = this -> b -> p; 
    double e_l  = this -> l -> e; 
    double e2l  = e_l*e_l; 

    double m2_l = this -> l -> m2;  
    double m2_b = this -> b -> m2; 

    double Lx = -1.0  / (2 * p_l);
    double Cx = -m2_l / (2 * p_l);
    double Ly = (1.0 / this -> _s) * (1.0/(2.0 * p_b) - this -> _c*Lx);
    double Cy = (1.0 / this -> _s) * ((m2_b - this -> mt2)/(2.0 * p_b) - this -> _c*Cx);

    double L_x1 = (1.0 - 1.0 / this -> om2())*Lx - (this -> w()/this -> om2())*Ly;
    double C_x1 = (1.0 - 1.0 / this -> om2())*Cx - (this -> w()/this -> om2())*Cy;
    double L_sub = Ly - this -> w()*Lx;
    double C_sub = Cy - this -> w()*Cx;
    
    double A2 =      this -> om2() * L_x1 * L_x1 - L_sub * L_sub  +  1.0 / (4 * e2l);
    double A1 = 2.0*(this -> om2() * L_x1 * C_x1 - L_sub * C_sub) + m2_l / (2 * e2l) - 1.0;
    double A0 =      this -> om2() * C_x1 * C_x1 - C_sub * C_sub  + pow(m2_l, 2) / (4 * e2l);
    double dsc = A1 * A1 - 4.0 * A2 * A0;
    *mw1_ = -1; *mw2_ = -1; 
    if (dsc < 0){return;}
    dsc = pow(dsc, 0.5); 
    
    double _mw1 = (-A1 + dsc) / (2.0 * A2); 
    double _mw2 = (-A1 - dsc) / (2.0 * A2); 
    *mw1_ = (_mw1 < 0) ? -1 : pow(_mw1, 0.5); 
    *mw2_ = (_mw2 < 0) ? -1 : pow(_mw2, 0.5); 
}

void nusol::r_mT(double* mt1_, double* mt2_){
   *mt1_ = -1; *mt2_ = -1; 
   double c = this -> _c; 
   double s = this -> _s; 
   double w_ = this -> w();
   double sx = this -> Sx(); 
   double o2 = this -> om2();
   double x0_ = this -> x0();
   
   double L_sy = -1.0 / (2.0 * s * this -> b -> p);
   double C_sy = (1.0/s) * ((this -> mw2 + this -> b -> m2)/(2.0 * this -> b -> p) - c*sx );
   
   double L_x1 = -w_*L_sy/o2;
   double C_x1 = sx - (sx + w_*C_sy)/o2;

   double L_sub = L_sy;
   double C_sub = C_sy - w_*sx;
   
   double B2 = o2 * L_x1 * L_x1 - L_sub * L_sub;
   double B1 = 2.0*(o2 * L_x1 * C_x1 - L_sub * C_sub);
   double B0 = o2*C_x1*C_x1 - C_sub*C_sub - (this -> mw2 - x0_*x0_ - this -> mw2 * (1 - this -> l -> b2));
   
   double disc = pow(B1, 2) - 4.0*B2*B0;
   if (disc < 0){return;}
   double sq = sqrt(disc);
   double mt2_s1 = (-B1 + sq) / (2.0 * B2);
   double mt2_s2 = (-B1 - sq) / (2.0 * B2);
   *mt1_ = (mt2_s1 < 0) ? -1 : sqrt(mt2_s1); 
   *mt2_ = (mt2_s2 < 0) ? -1 : sqrt(mt2_s2); 
}

void nusol::Z_mW(double* mw1_, double* mw2_){
    double w_ = (this -> l -> b / this -> b -> b - this -> _c) / this -> _s;
    double o2 = this -> om2();

    double E1 = -1 / (2 * this -> l -> e); 
    double P1 =  1 / (2 * this -> b -> p); 
    double E0 = this -> l -> m2 / (2 * this -> l -> e); 
    double sx = E0 - this -> l -> m2 / this -> l -> e; 
    double P0 = (this -> b -> m2 - this -> mt2) / (2 * this -> b -> p); 

    double sy0 = (P0 - this -> _c* sx) / this -> _s;  
    double sy1 = (P1 - this -> _c* E1) / this -> _s;  
    double X0 = sx * (1 - 1/o2) - (this -> w() * sy0) / o2; 
    double X1 = E1 * (1 - 1/o2) - (this -> w() * sy1) / o2; 
    
    double D0 = sy0 - this -> w() * sx; 
    double D1 = sy1 - this -> w() * E1; 
    
    double A = o2 * X1*X1 - D1*D1 + E1*E1; 
    double B = o2 * X0 * X1 - D0 * D1 + E0 * E1 - this -> l -> b2; 
   
    double v_c = -B / A; 
    double v_i = -B / (3.0 * A); 
    *mw1_ = (v_c < 0) ? -1 : sqrt(v_c); 
    *mw2_ = (v_i < 0) ? -1 : sqrt(v_i); 
}

void nusol::Z_mT(double* mt1_, double* mt2_, double _mw){
    double w_ = (this -> l -> b / this -> b -> b - this -> _c) / this -> _s;
    double o2 = this -> om2();
    double mW2 = _mw*_mw; 

    double x0_ = -(mW2 - this -> l -> m2) / (2 * this -> l -> e); 
    double cnx =   mW2 - x0_*x0_ - mW2 * (1 - this -> l -> b2); 
    double sx_ = x0_ - this -> l -> e * (1 - this -> l -> b2); 

    double A_sy = -1 / (2 * this -> b -> p * this -> _s); 
    double B_sy = ((mW2 + this -> b -> m2) / (2 * this -> b -> p) - this -> _c * sx_) / this -> _s; 

    double A_x1 = -(w_ * A_sy) / o2; 
    double B_x1 = ((o2 - 1) * sx_ - w_ * B_sy) / o2; 
    double A =  o2 * A_x1 * A_x1 - A_sy*A_sy; 
    double B = (o2 * A_x1 * B_x1 - A_sy * (B_sy - w_ * sx_)); 
    double C =  o2 * B_x1 * B_x1 - pow(B_sy - w_ * sx_, 2) - cnx - this -> Z2(); 
    double disc = B*B - A * C; 

    double root1 = (-B + sqrt(disc)) / A; 
    double root2 = (-B - sqrt(disc)) / A; 
    *mt1_ = (root1 < 0 || disc < 0) ? -1 : sqrt(root1); 
    *mt2_ = (root2 < 0 || disc < 0) ? -1 : sqrt(root2); 
}

void nusol::update(double mt_, double mw_){
    this -> mt = (mt_ > 0) ? mt_ : this -> mt; 
    this -> mt2 = this -> mt * this -> mt; 

    this -> mw = (mw_ > 0) ? mw_ : this -> mw; 
    this -> mw2 = this -> mw * this -> mw; 
    this -> flush(); 
}

void nusol::misc(){
    std::cout << "--------- Neutrino ------------" << std::endl;
    std::cout << "Sx:      " << this -> Sx()       << std::endl;
    std::cout << "dSx_dmW: " << this -> dSx_dmW()  << std::endl;
    std::cout << "Sy:      " << this -> Sy()       << std::endl;         
    std::cout << "dSy_dmW: " << this -> dSy_dmW()  << std::endl;  
    std::cout << "dSy_dmT: " << this -> dSy_dmT()  << std::endl;  
    std::cout << "w:       " << this -> w()        << std::endl;        
    std::cout << "w2:      " << this -> w2()       << std::endl;       
    std::cout << "om2:     " << this -> om2()      << std::endl;      
    std::cout << "Z:       " << this -> Z()        << std::endl;        
    std::cout << "Z2:      " << this -> Z2()       << std::endl;       
    std::cout << "dZ_dmT:  " << this -> dZ_dmT()   << std::endl;   
    std::cout << "dZ_dmW:  " << this -> dZ_dmW()   << std::endl;   
    std::cout << "x1:      " << this -> x1()       << std::endl;       
    std::cout << "dx1_dmW: " << this -> dx1_dmW()  << std::endl;  
    std::cout << "dx1_dmT: " << this -> dx1_dmT()  << std::endl;  
    std::cout << "y1:      " << this -> y1()       << std::endl;       
    std::cout << "dy1_dmW: " << this -> dy1_dmW()  << std::endl;  
    std::cout << "dy1_dmT: " << this -> dy1_dmT()  << std::endl;  

    double A, B, C; 
    this -> Z2(&A, &B, &C); 
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



