#include "multisol/conuix.h"

long double conuic::Z2(long double sx, long double sy){
    long double out = 0; 
    out += this -> state.Z2.a * sx * sx;
    out += this -> state.Z2.b * sx * sy;
    out += this -> state.Z2.c * sy * sy;
    out += this -> state.Z2.d * sx;
    out += this -> state.Z2.e; 
    return out; 
}

long double conuic::Sx(long double ct, long double st, long double Z){
    return Z * (this -> state.Sx.a * ct + this -> state.Sx.b * st) + this -> state.Sx.c; 
}

long double conuic::Sy(long double ct, long double st, long double Z){
    return Z * (this -> state.Sy.a * ct + this -> state.Sy.b * st) + this -> state.Sy.c; 
}

long double conuic::Sx(long double mw, long double mt){
    long double b_ = this -> state.lep.c; 
    long double x0 = -(mw * mw - this -> state.lep.a) / this -> state.lep.b; 
    return (x0 * b_ - this -> state.lep.d * (1 - b_ * b_)) / (b_ * b_); 
}

long double conuic::Sy(long double mw, long double mt){
    long double x0p = -(mt * mt - mw * mw - this -> state.jet.a)/ this -> state.jet.b; 
    return (x0p - this -> _cth * this -> Sx(mw, mt)) / this -> _sth; 
}

long double conuic::x1(long double ct, long double st, long double Z){
    return Z * (this -> state.x1.a * ct + this -> state.x1.b * st); 
}

long double conuic::y1(long double ct, long double st, long double Z){
    return Z * (this -> state.y1.a * ct + this -> state.y1.b * st); 
}


coef_t conuic::get_tauZ(long double sx, long double sy){
    long double f  = this -> state.tZ.a;
    long double kx = this -> state.tZ.b;
    long double ky = this -> state.tZ.c; 

    long double a = (sy + ky) * this -> _cpsi - (sx + kx) * this -> _spsi; 
    long double b = (sx + kx) * this -> _cpsi + (sy + ky) * this -> _spsi; 
    long double t = f * (a / b);

    coef_t r;
    r.a = t; 
    r.b = (std::fabs(t) >= 1) ? -2 : std::atanh(t); 
    r.c = f * this -> _cpsi * std::cosh(r.b);
    r.c = (sx + kx) / (r.c - this -> _spsi * std::sinh(r.b)); 
    return r;
}

coef_t conuic::masses(long double Z, long double t){
    this -> hyper(t); 
    long double sx_ = this -> Sx(this -> _ct, this -> _st, Z); 
    long double sy_ = this -> Sy(this -> _ct, this -> _st, Z); 
    long double sxy = this -> _sth * sy_ + this -> _cth * sx_; 

    coef_t out; 
    out.a_cplx = this -> state.mass.a + this -> state.mass.b * sx_; 
    out.b_cplx = this -> state.mass.c + this -> state.mass.d * sxy + out.a_cplx; 
    out.a_cplx = std::sqrt(out.a_cplx);
    out.b_cplx = std::sqrt(out.b_cplx); 
    return out; 
}

coef_t conuic::mass_line(long double mW, long double mT){
    long double y = this -> state.line.c * mT * mT + this -> state.line.d * mW * mW + this -> state.line.e;
    long double x = this -> state.line.a * mW * mW + this -> state.line.b;
    
    long double uy = (y * this -> state.Sx.a - x * this -> state.Sy.a);
    long double ux = (x * this -> state.Sy.b - y * this -> state.Sx.b);
    long double t  = uy / ux;  

    std::complex<long double> d1 = this -> state.Sx.a + this -> state.Sx.b * t;
    std::complex<long double> d2 = this -> state.Sy.a + this -> state.Sy.b * t;
    std::complex<long double> st = std::sqrt(std::complex<long double>(1.0L - t * t));

    coef_t out; 
    out.a_cplx = (std::fabs(d1) > std::fabs(d2)) ? (x * st / d1) : (y * st / d2);
    out.b_cplx = std::atanh(std::complex<long double>(t, 0.0L)); 
    return out;
}


matrix_t conuic::H_matrix(long double ct, long double st){
    return (*this -> state.HTS) * st + (*this -> state.HTC) * ct + (*this -> state.HTX); 
}

matrix_t conuic::H_perp(long double ct, long double st, long double Z){
    matrix_t H = this -> H_matrix(ct, st) * Z;
    H.at(2, 0) = 0.0L; 
    H.at(2, 1) = 0.0L;
    H.at(2, 2) = 1.0L;
    return H; 
}

matrix_t conuic::N(long double ct, long double st, long double Z, bool full){
    matrix_t H = (full ? this -> H_matrix(ct, st) * Z : this -> H_perp(ct, st, Z)).inv(); 
    return H.T().dot(circle()).dot(H); 
}

matrix_t conuic::H_tilde(long double ct, long double st){
    return (*this -> state.HBS) * st + (*this -> state.HBC) * ct + (*this -> state.HBX); 
}

matrix_t conuic::K(long double ct, long double st, long double Z, bool full){
    matrix_t kx = (full ? this -> H_matrix(ct, st) * Z : this -> H_perp(ct, st, Z)).inv(); 
    return this -> H_matrix(ct, st).dot(kx); 
}

matrix_t conuic::Nu(const matrix_t nu, long double Z, bool full){
    matrix_t kx = this -> K(this -> _ct, this -> _st, 1.0, full).dot(nu) * Z; 
    return this -> make_neutrino(&kx); 
} 

coef_t conuic::root_mW(long double mT){
    long double lx = this -> state.r_mW.a;
    long double ly = this -> state.r_mW.c; 
    long double cx = this -> state.r_mW.b;
    long double cy = this -> state.r_mW.d + this -> state.r_mW.e * mT * mT; 
    long double fx = 2.0L * this -> state.lep.a / this -> state.lep.b; 
    
    std::complex<long double> LxC = lx * cx + ly * cy;
    std::complex<long double> LL  = lx * lx + ly * ly; 
    
    coef_t out;
    out.a_cplx = std::sqrt((- LxC + fx * std::sqrt( LL )) / LL);
    out.b_cplx = std::sqrt((- LxC - fx * std::sqrt( LL )) / LL);
    return out; 
}

long double conuic::PL(long double z, long double l){
    long double fx = this -> _o * this -> _st + this -> mobius[3] * this -> _ct; 
    long double zi = z / this -> _o; 

    long double _a =   l * l * l; 
    long double _b = - l * l * zi; 
    long double _c =   (l * z * zi) * this -> _cpsi * fx; 
    long double _d = - (z * z * zi) * (1.0L / this -> _cpsi) * this -> _st;
    return _a + _b + _c + _d; 
}

coef_t conuic::dPdZ0(long double z){
    long double sc = 1.0L / this -> _cpsi; 
    std::complex<long double> _a = this -> _o * this -> _st - this -> beta * this -> _tpsi * this -> _ct;
    std::complex<long double> _b = std::sqrt(_a * _a - 3 * sc * sc * sc * this -> _st); 
    
    coef_t out; 
    out.a_cplx = (_a + _b) * z * this -> _cpsi; 
    out.b_cplx = (_a - _b) * z * this -> _cpsi; 
    return out; 
}


long double conuic::dPl0(){
    long double ap = this -> alpha_p(this -> _tt); 
    long double an = this -> alpha_m(this -> _tt); 
    return this -> a_ * std::sqrt(1 - this -> _tt * this -> _tt) * an - ap / an; 
}

long double conuic::dPdt(long double z, long double l){
    long double fx =  (this -> _o *  this -> _ct + this -> mobius[3] * this -> _st); 
    long double _a =   (l * z * z /  this -> _o) * fx * this -> _cpsi; 
    long double _b = - (z * z * z) * this -> _ct / (this -> _o * this -> _cpsi); 
    return _a + _b; 
}

long double conuic::dPdtL0(long double z){
    return (z * this -> _o / (this -> _cpsi * this -> _cpsi)) * 1.0L / this -> alpha_m(this -> _tt); 
}


