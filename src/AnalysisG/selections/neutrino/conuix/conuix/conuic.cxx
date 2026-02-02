#include <templates/particle_template.h>
#include <conuix/conuic.h>

conuic::conuic(particle_template* jet, particle_template* lep){
    this -> _jet = jet;
    this -> _lep = lep; 
    this -> cache = new atomics_t(this -> _jet, this -> _lep); 
    this -> tstar = this -> cache -> Mobius.tstar; 
    this -> error = this -> cache -> Mobius.error; 
    this -> converged = this -> cache -> Mobius.converged;

    this -> vstar = matrix_t(3, 1); 
    this -> cache -> eigenvector(this -> cache -> Mobius.tstar, &this -> vstar, &this -> theta);     
}

conuic::~conuic(){
    this -> _jet = nullptr; 
    this -> _lep = nullptr; 
    if (!this -> cache){return;}
    delete this -> cache; 
    this -> cache = nullptr; 
}

void conuic::debug(){
    std::string hx = std::string(this -> _jet -> hash) + "-" + std::string(this -> _lep -> hash);
    if (hx != "0x5a4ac8ead1f7d952-0x1dbf9ed01fdef73b"){return;}
    this -> cache -> debug_mode(this -> _jet, this -> _lep); 
    abort(); 
}

long double conuic::Z2(long double sx, long double sy){
    return this -> cache -> pencil.Z2(sx, sy); 
}

long double conuic::Sx(long double t, long double Z){
    return this -> cache -> Sx.Sx(t, Z); 
}

long double conuic::Sy(long double t, long double Z){
    return this -> cache -> Sy.Sy(t, Z); 
}

long double conuic::x1(long double t, long double Z){
    return this -> cache -> x1(Z, t); 
}

long double conuic::y1(long double t, long double Z){
    return this -> cache -> y1(Z, t); 
}

long double conuic::P(long double l, long double t, long double Z){
    return this -> cache -> Mobius.P(Z, l, t); 
}

long double conuic::dPdt(long double l, long double t, long double Z){
    return this -> cache -> Mobius.dPdt(Z, l, t); 
}

long double conuic::dPdtL0(long double t, long double Z){
    return this -> cache -> Mobius.dPdtL0(Z, t); 
}

long double conuic::dPl0(long double t){
    return this -> cache -> Mobius.dPl0(t, true); 
}

bool conuic::get_TauZ(long double sx, long double sy, long double* z, long double* t){
    return this -> cache -> GetTauZ(sx, sy, z, t); 
}

matrix_t conuic::Hmatrix(long double t, long double Z){
    return this -> cache -> H_Matrix.H_Matrix(t, Z); 
}

matrix_t conuic::Nmatrix(long double t, long double Z){
    matrix_t Hxi = this -> Hmatrix(t, Z).inv(); 
    matrix_t HxT = Hxi.T(); 
    matrix_t Cr  = matrix_t(3, 3); 
    Cr.at(0,0) =  1.0L; 
    Cr.at(1,1) =  1.0L; 
    Cr.at(2,2) = -1.0L; 
    return HxT.dot(Cr).dot(Hxi); 
}

bool conuic::mass_line(long double mW, long double mT, std::complex<long double>* z_, long double* t_){
    long double lm = this -> cache -> lp.mass;
    long double lb = this -> cache -> lp.beta;
    long double le = this -> cache -> lp.energy; 

    long double jm = this -> cache -> jt.mass;
    long double jb = this -> cache -> jt.beta;
    long double je = this -> cache -> jt.energy; 

    long double cs = this -> cache -> base.cos;
    long double ss = this -> cache -> base.sin; 

    long double a = -(mW * mW + lm * lm) / (2 * le * lb);
    long double b = -(mT * mT - mW * mW - jm * jm) / (2 * je * jb);

    long double a0 =   this -> cache -> base.o * this -> cache -> base.cpsi / lb; 
    long double b0 = - this -> cache -> base.spsi; 
    long double c0 = - lm * lm / (le * lb); 
    long double d0 =   this -> cache -> base.o * this -> cache -> base.spsi / lb; 
    long double e0 =   this -> cache -> base.cpsi; 
    long double f0 = - this -> cache -> base.tpsi * le / lb; 

    long double x = a * (a0 * cs + d0 * ss) - (b - c0 * cs - f0 * ss) * a0;
    long double y = a * (b0 * cs + e0 * ss) - (b - c0 * cs - f0 * ss) * b0;
    //std::complex<long double> zeta = 0.5L * std::log( std::complex<long double>(x + y, 0.0) / std::complex<long double>(y - x, 0.0) ); 
    //std::complex<long double> tcmx = std::tanh(zeta); 
    long double t = std::sin(std::atan2(x, y)); 
    
    std::complex<long double> z = -((mW * mW + lm * lm) / ( 2 * le * lb) + c0) * std::sqrt((1.0L - t*t)) / (a0 + b0 * t); 
    if (z_){*z_ = std::abs(z);}
    if (t_){*t_ = t;}
    if (std::fabs(x / y) >= 1){return false;}
    return true;
}

void conuic::dPdZ0(long double z, long double t, std::complex<long double>* l1, std::complex<long double>* l2){
    long double o_ = this -> cache -> base.o; 
    long double bl = this -> cache -> lp.beta; 
    long double ts = this -> cache -> base.tpsi;
    long double sc = 1.0L / this -> cache -> base.cpsi; 
    std::complex<long double> a_ = o_ * std::sinh(t) - bl * ts * std::cosh(t);
    std::complex<long double> b_ = std::sqrt(a_ * a_ - 3 * sc * sc * sc * std::sinh(t)); 
    std::complex<long double> rp = (a_ + b_) * z * this -> cache -> base.cpsi; 
    std::complex<long double> rm = (a_ - b_) * z * this -> cache -> base.cpsi; 
    *l1 = rp; *l2 = rm; 
}













