#include "multisol/cache.h"
#include <cmath>

cache::cache(particle_template* jt, particle_template* lp){this -> jet = jt; this -> lep = lp;}
cache::~cache(){
    this -> dSafe(&this -> RT ); this -> dSafe(&this -> NU); 
    this -> dSafe(&this -> HBX); this -> dSafe(&this -> HTX);
    this -> dSafe(&this -> HBC); this -> dSafe(&this -> HTC);
    this -> dSafe(&this -> HBS); this -> dSafe(&this -> HTS); 
    this -> dSafe(&this -> vlep); this -> dSafe(&this -> vjet); 
}

void cache::hyper(long double tau){
    this -> _tt = std::tanh(tau); 
    this -> _ct = std::cosh(tau);
    this -> _st = std::sinh(tau); 
}

long double cache::alpha_p(long double u){
    return this -> mobius[0] + this -> mobius[1] * u;
}

long double cache::alpha_m(long double u){
    return this -> mobius[2] + this -> mobius[3] * u;
}


matrix_t cache::make_neutrino(const matrix_t* v){
    long double e = 0; 
    for (int x(0); x < 3; ++x){e += v -> at(x, 0) * v -> at(x, 0);}
    matrix_t v_(1, 4); 
    v_.at(0, 0) = v -> at(0, 0); 
    v_.at(0, 1) = v -> at(1, 0);
    v_.at(0, 2) = v -> at(2, 0);
    v_.at(0, 3) = std::sqrt(e); 
    return v_; 
}

matrix_t cache::make_top(const matrix_t* v){
    return *v + this -> vjet -> clone() + this -> vlep -> clone(); 
}

matrix_t cache::make_w(const matrix_t* v){
    return *v + this -> vlep -> clone(); 
}

std::complex<long double> cache::get_mass(const matrix_t* v){
    long double px = v -> at(0, 0);
    long double py = v -> at(0, 1);
    long double pz = v -> at(0, 2); 
    long double e  = v -> at(0, 3); 
    return std::sqrt(std::complex<long double>(e*e - px*px + py*py + pz*pz)); 
}

cache_t cache::init(){

    // =========== kinematics ========= //
    // neutrino 
    long double nm2 = 0;  
     
    // lepton
    long double lpx = double(this -> lep -> px); 
    long double lpy = double(this -> lep -> py); 
    long double lpz = double(this -> lep -> pz); 
    long double lm  = double(std::fabs(this -> lep -> mass));
    long double le  = double(this -> lep -> e); 

    long double lb  = double(this -> lep -> beta); 
    long double lp  = le * lb; 

    long double lb2 = lb * lb; 
    long double lm2 = lm * lm; 
    this -> hash = this -> jet -> hash + this -> lep -> hash; 


    // jet
    long double bpx = double(this -> jet -> px); 
    long double bpy = double(this -> jet -> py); 
    long double bpz = double(this -> jet -> pz); 
    long double bm  = double(std::fabs(this -> jet -> mass));
    long double be  = double(this -> jet -> e); 

    long double bb  = double(this -> jet -> beta); 
    long double bp  = be * bb; 
    long double bm2 = bm * bm; 

    cache_t out; 
    out.RT = this -> get_rotation(lpx, lpy, lpz, lp, bpx, bpy, bpz);
    this -> vlep = new matrix_t(1, 4);
    this -> vlep -> at(0, 0) = lpx;
    this -> vlep -> at(0, 1) = lpy;
    this -> vlep -> at(0, 2) = lpz;
    this -> vlep -> at(0, 3) = le; 

    this -> vjet = new matrix_t(1, 4);
    this -> vjet -> at(0, 0) = bpx;
    this -> vjet -> at(0, 1) = bpy;
    this -> vjet -> at(0, 2) = bpz;
    this -> vjet -> at(0, 3) = be; 

    // -------------- base relation ----------------- //
    long double rbl = lb / bb; 

    long double cth = (bpx * lpx + bpy * lpy + bpz * lpz) / (bp * lp); 
    long double sth = std::sqrt(1 - cth*cth); 

    long double w   = (rbl - cth)/sth; 
    long double w2  = w*w; 

    long double o2 = w2 + 1 - lb2; 
    long double o  = std::sqrt(o2);
     
    long double tpsi = w; 
    long double cpsi = 1.0L / std::sqrt(1 + w2);
    long double spsi = w * cpsi;
    long double olb  = o / lb; 
        
    this -> _cpsi = cpsi;
    this -> _spsi = spsi;
    this -> _tpsi = tpsi;

    this -> _sth = sth; 
    this -> _cth = cth; 
    this -> _o   = o; 


    this -> get_hmatrix(tpsi, cpsi, spsi, o, lb);
    out.HBX = this -> HBX; out.HTX = this -> HTX;
    out.HBC = this -> HBC; out.HTC = this -> HTC;
    out.HBS = this -> HBS; out.HTS = this -> HTS;

    out.lep.a = lm2;
    out.lep.b = 2.0L * le; 
    out.lep.c = lb; 
    out.lep.d = lp; 
    
    out.jet.a = bm2; 
    out.jet.b = 2.0L * be; 


    // Z^2 = a Sx^2 + b Sx Sy + c Sy^2 + d Sx + e 
    out.Z2.a = (1.0L - o2) / o2;
    out.Z2.b =    2.0L * w / o2;
    out.Z2.c =   (w2 - o2) / o2; 
    out.Z2.d = 2.0L * lp; 
    out.Z2.e = lm2 - nm2; 

    // Sx(tau) = Z [ (Omega / beta_mu) cos(psi) cosh(tau) - sin(psi) sinh(tau)] - m^2_mu / (E_mu beta_mu)
    out.Sx.a = olb * cpsi;
    out.Sx.b =     - spsi; 
    out.Sx.c = - lm2 / lp; 

    // Sy(tau) = Z [(sin(psi) / beta_mu) Omega cosh(tau) + cos(psi) sinh(tau)] - tan(psi) E_mu / beta_mu
    out.Sy.a =  olb * spsi;
    out.Sy.b =        cpsi; 
    out.Sy.c = - (le / lb) * tpsi; 

    // x1(tau) 
    out.x1.a = (1.0L / olb) * lb * le * cpsi; // cosh(tau)
    out.x1.b =                lb * le * spsi; // sin(tau)

    // y1(tau) 
    out.y1.a = (1.0L / olb) * spsi; // sin(tau)
    out.y1.b =              - cpsi; 

    // GettauZ
    out.tZ.a = olb; 
    out.tZ.b = lm2 / lp;
    out.tZ.c = (le / lb) * tpsi; 

    // masses
    out.mass.a = - lm2; 
    out.mass.b = - 2 * lp; 
    out.mass.c =   bm2;
    out.mass.d = - 2 * bp;  

    // Z and tau given mW and mT
    out.line.a = -0.5L / lp;                           
    out.line.b =  0.5L * (lm2 / lp);                      
    out.line.c = -0.5L / (bp * sth);                       
    out.line.d =  0.5L / (bp * sth) + cth * 0.5L / (lp * sth);  
    out.line.e =  0.5L * (bm2 / (bp * sth)) + lm2 * cth * 0.5L / (lp * sth) + (le / lb) * tpsi;   

    // r_mW 
    out.r_mW.a = - 1.0L / (2.0L * lp);
    out.r_mW.b = -  lm2 / (2.0L * lp); 
    out.r_mW.c =  (1.0L / (2.0L * sth)) * (1.0L / bp + cth/lp); 
    out.r_mW.d =  (1.0L / (2.0L * sth)) * (bm2 / bp + lm2 * cth / lp); 
    out.r_mW.e = -(1.0L / (2.0L * sth * bp)); 

    // ------------------ Polynomial stuff ------------------- //
    std::complex<long double> _f = tpsi * (o - lb);
    std::complex<long double> _c = std::sqrt(_f*_f + 4 * o * lb * (1 + tpsi * tpsi)); 
    this -> e_val[0] = (_f + _c) / 2.0L;
    this -> e_val[1] = (_f - _c) / 2.0L; 
    this -> e_vec[0] = 1.0L;
    this -> e_vec[1] = 1.0L;
    this -> e_vec[2] = (this -> e_val[0] + lb * tpsi) / o;
    this -> e_vec[3] = (this -> e_val[1] + lb * tpsi) / o; 

    _c = 4 * lb * o * (1 + tpsi * tpsi); 
    this -> kfactor  = (_f - std::sqrt(_f * _f + _c)) / (_f + std::sqrt(_f * _f + _c)); 
    this -> midpoint = -tpsi * (o + lb) / (2.0L * lb);  

    _f = - (o + lb) * tpsi; 
    _c = std::sqrt(_f * _f + 4 * lb * o);
    this -> fixed[0] = (_f + _c) / (2.0L * lb);
    this -> fixed[1] = (_f - _c) / (2.0L * lb);

    this -> poles[0] =   (o / lb) * (1.0L / tpsi); 
    this -> poles[1] = - (o / lb) * tpsi; 
    this -> mobius[0] = o * tpsi; 
    this -> mobius[1] = lb;
    this -> mobius[2] = o;
    this -> mobius[3] = -tpsi * lb; 

    this -> sym_axis = 0.5L * std::atanh(std::complex<long double>( o * lb / ((o * o + lb * lb) * spsi * cpsi))); 

    this -> a_   = lb * std::pow(_cpsi, 3); 
    this -> beta = lb; 
    this -> get_dPL0dT0(); 

    return out; 
}


void cache::get_dPL0dT0(){
    auto PL =[this](long double z, long double l){
        long double fx = this -> _o * this -> _st + this -> mobius[3] * this -> _ct; 
        long double zi = z / this -> _o; 

        long double _a =   l * l * l; 
        long double _b = - l * l * zi; 
        long double _c =   (l * z * zi) * this -> _cpsi * fx; 
        long double _d = - (z * z * zi) * (1.0L / this -> _cpsi) * this -> _st;
        return _a + _b + _c + _d; 
    };

    auto dPdZ0 =[this](long double z){
        long double sc = 1.0L / this -> _cpsi; 
        long double _a = this -> _o * this -> _st - this -> beta * this -> _tpsi * this -> _ct;
        return _a * _a - 3 * sc * sc * sc * this -> _st; 
    };

    auto dPdtL0 =[this](long double z){
        return (z * this -> _o / (this -> _cpsi * this -> _cpsi)) * 1.0L / this -> alpha_m(this -> _tt); 
    };


    long double tx_ = std::pow(this -> _cpsi, -3); 
    long double bx_ = 2 * this -> beta * this -> beta * this -> _tpsi; 
    long double ob = this -> _o * this -> _o + this -> beta * this -> beta; 
   
    int idx = 0;  
    int zsign = 0; 
    int tsign = 0; 
    bool fone  = false; 
    this -> taustar[0] = this -> midpoint; 
    for (int i(0); i < 10000000; ++i){
        long double p_ = (5000000 -1 - i) * 1.0 / 5000000L; 
        this -> _tt = p_; 
        this -> _st = p_ / std::sqrt(1 - p_); 
        this -> _ct = std::sqrt(1 + this -> _st * this -> _st); 


        long double x   = this -> taustar[0]; 
        long double bx  = this -> alpha_m(x); 
        long double b2x = bx * bx; 
        long double sq = std::sqrt(1 - x * x); 
       
        // plus branch 
        long double ap = this -> alpha_p(p_); 
        long double am = this -> alpha_m(p_); 

        long double dp = dPdZ0(1); 
        long double pl = PL(1, dPdtL0(1)); 
        bool trig = (dp < 0 && zsign > 0 || dp > 0 && zsign < 0); 
        trig     += (pl < 0 && tsign > 0 || pl > 0 && tsign < 0); 
        zsign = -1 * (dp < 0) + (dp > 0);
        tsign = -1 * (pl < 0) + (pl > 0);
        this -> taupts[idx] = trig *  p_;
        idx += trig; 
        std::cout << " " << p_ << " " << trig << " " << pl << " " << dp << " " << zsign << " " << tsign << std::endl;

        if (p_ < -1){abort(); break;}
        continue;

        this -> taustar[1] =   this -> beta * sq * b2x - tx_ * this -> alpha_p(x);   
        this -> taustar[2] = - this -> beta * ( (x / sq) * b2x  + tx_ ) - bx_ * sq * bx; 
        this -> taustar[0] = x - this -> taustar[1] / this -> taustar[2];

        bool f = this -> taustar[0] < this -> poles[0]; 
        f     *= this -> taustar[0] > this -> poles[1]; 
        f     *= !std::isnan(std::fabs(x)); 
        if (!f){
            this -> converged = false; 
            if (fone){break;}
            this -> taustar[0] = 0;
            fone = true; 
            continue;
        }

        if (std::fabs(this -> taustar[2]) < 10e-15){break;}
        if (std::fabs(this -> taustar[1]) > 10e-11){continue;}
        this -> converged = true; 
        break; 
    }

    this -> taustar[2] = (this -> converged) ? std::atanh(this -> taustar[0]) : -99; 
    if (!this -> converged){return;}
    if (this -> NU){return;}

    long double  t = this -> taustar[2]; 
    long double t_ = this -> _tpsi; 
    long double a1 = this -> _cpsi * this -> beta * std::cosh(t) + this -> _spsi * this -> _o * std::sinh(t);
    long double mu = this -> _o    * (1 + t_*t_) / (this -> _o - this -> beta * t_ * std::tanh(t));

    matrix_t nu = matrix_t(3, 1); 
    nu.at(0,0) = a1 / (mu - 1);
    nu.at(1,0) = mu / this -> _o; 
    nu.at(2,0) = 1.0L; 
    this -> NU = new matrix_t(this -> RT -> dot(nu)); 
    this -> taustar[3] = std::atan2(this -> NU -> at(1,0), this -> NU -> at(0,0)); 
}

void cache::get_hmatrix(long double tpsi, long double cpsi, long double spsi, long double o, long double lb){
    // to get HMatrix -> i.e. the transformation matrix of the ellipse.

    // ----------- build static matrices ----------- //
    this -> HBX = new matrix_t(3, 3); 

    this -> HBX -> at(0, 0) = 1.0L / o; 
    this -> HBX -> at(1, 0) = tpsi / o; 
    this -> HBX -> at(2, 0) = 0.0L;

    this -> HBX -> at(0, 1) = 0.0L; 
    this -> HBX -> at(1, 1) = 0.0L;
    this -> HBX -> at(2, 1) = 1.0L;

    this -> HBX -> at(0, 2) = 0.0L;
    this -> HBX -> at(1, 2) = 0.0L;
    this -> HBX -> at(2, 2) = 0.0L; 
    this -> HTX = new matrix_t(this -> RT -> dot(*this -> HBX)); 

    // .......... cosh matrix.............. //
    this -> HBC = new matrix_t(3, 3); 
    this -> HBC -> at(0, 0) = 0.0L; 
    this -> HBC -> at(1, 0) = 0.0L; 
    this -> HBC -> at(2, 0) = 0.0L;

    this -> HBC -> at(0, 1) = 0.0L; 
    this -> HBC -> at(1, 1) = 0.0L;
    this -> HBC -> at(2, 1) = 0.0L;

    this -> HBC -> at(0, 2) = lb * cpsi / o;
    this -> HBC -> at(1, 2) = lb * spsi / o;
    this -> HBC -> at(2, 2) = 0.0L; 
    this -> HTC = new matrix_t(this -> RT -> dot(*this -> HBC));

    // .......... sinh matrix.............. //
    this -> HBS = new matrix_t(3, 3); 
    this -> HBS -> at(0, 0) = 0.0L; 
    this -> HBS -> at(1, 0) = 0.0L; 
    this -> HBS -> at(2, 0) = 0.0L;

    this -> HBS -> at(0, 1) = 0.0L; 
    this -> HBS -> at(1, 1) = 0.0L;
    this -> HBS -> at(2, 1) = 0.0L;

    this -> HBS -> at(0, 2) =   spsi;
    this -> HBS -> at(1, 2) = - cpsi;
    this -> HBS -> at(2, 2) = 0.0L; 
    this -> HTS = new matrix_t(this -> RT -> dot(*this -> HBS)); 
}

const matrix_t* cache::get_rotation(
    long double lpx, long double lpy, long double lpz, long double lp,
    long double bpx, long double bpy, long double bpz
){
    if (this -> RT){return this -> RT;}
    long double phi   = std::atan2(lpy, lpx); 
    long double theta = 0.5 * M_PI - std::acos(lpz / lp );  
    
    matrix_t vec = matrix_t(3, 1); 
    vec.at(0, 0) = bpx; 
    vec.at(1, 0) = bpy; 
    vec.at(2, 0) = bpz; 
    
    matrix_t Rz(3, 3);
    Rz.at(0, 0) =  std::cos(-phi); 
    Rz.at(0, 1) = -std::sin(-phi); 
    Rz.at(2, 2) = 1;
    Rz.at(1, 0) =  std::sin(-phi);
    Rz.at(1, 1) =  std::cos(-phi);
    
    matrix_t Ry(3, 3); 
    Ry.at(0, 0) =  std::cos(theta); 
    Ry.at(0, 2) =  std::sin(theta); 
    Ry.at(1, 1) = 1;
    Ry.at(2, 0) = -std::sin(theta); 
    Ry.at(2, 2) =  std::cos(theta);

    matrix_t b_p = Ry.dot(Rz.dot(vec)); 
    long double alpha = -std::atan2(b_p.at(2, 0), b_p.at(1, 0));

    matrix_t Rx(3, 3);
    Rx.at(0, 0) = 1; 
    Rx.at(1, 1) =  std::cos(alpha); 
    Rx.at(1, 2) = -std::sin(alpha);
    Rx.at(2, 1) =  std::sin(alpha); 
    Rx.at(2, 2) =  std::cos(alpha);
    this -> RT = new matrix_t(Rz.T().dot(Ry.T().dot(Rx.T()))); 
    return this -> RT; 
}


