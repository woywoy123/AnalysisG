#include <templates/particle_template.h>
#include <conuix/struct.h> 
#include <conuix/htilde.h>

atomics_t::atomics_t(particle_template* jet, particle_template* lep, double m_nu){
    this -> nu.mass = m_nu;

    // ------------ get all the kinematics --------- //
    Conuix::get_kinematic(lep, &this -> lp); 
    Conuix::get_kinematic(jet, &this -> jt); 

    this -> base.cos = Conuix::cos_theta(jet, lep); 
    this -> base.sin = std::sqrt(1 - this -> base.cos * this -> base.cos); 

    Conuix::get_base(&this -> jt, &this -> lp, &this -> base); 
    Conuix::get_rotation(jet, lep, &this -> rotation); 

    Conuix::get_psi_theta_mapping(&this -> base, &this -> psi_theta);
    Conuix::get_pencil(&this -> lp, &this -> nu, &this -> base, &this -> pencil); 

    Conuix::get_sx(&this -> base, &this -> Sx);
    Conuix::get_sy(&this -> base, &this -> Sy); 

    Conuix::get_hmatrix(&this -> base, &this -> rotation, &this -> H_Matrix); 
}

atomics_t::~atomics_t(){

}

void Conuix::debug::print(int p){}
void Conuix::debug::variable(std::string name, long double val){
    std::cout << std::fixed << std::setprecision(this -> prec);
    std::cout << name << ": " << val << " | " << std::endl;
}

void Conuix::kinematic_t::print(int p){
    this -> prec = p; 
    this -> variable("kinematic::beta", this -> beta); 
    this -> variable("kinematic::mass", this -> mass);
    this -> variable("kinematic::energy", this -> energy);  
}

void Conuix::rotation_t::print(int p){
    this -> prec = p; 
    this -> variable("rotation::phi"  , this -> phi); 
    this -> variable("rotation::theta", this -> theta);
    this -> vec.print(p);
    this -> R_T.print(p); 
}

void Conuix::base_t::print(int p){
    this -> prec = p; 
    this -> variable("base::cos"    , this -> cos); 
    this -> variable("base::sin"    , this -> sin);
    this -> variable("base::omega"  , this -> w);
    this -> variable("base::Omega"  , this -> o);
    this -> variable("base::omega2" , this -> w2);
    this -> variable("base::Omega2" , this -> o2);
    this -> variable("base::beta"   , this -> beta);
    this -> variable("base::mass"   , this -> mass);
    this -> variable("base::energy" , this -> E);
    this -> variable("base::tan(psi)", this -> tpsi);
    this -> variable("base::cos(psi)", this -> cpsi);
    this -> variable("base::sin(psi)", this -> spsi);
}

void Conuix::pencil_t::print(int p){
    this -> prec = p; 
    this -> variable("pencil::a" , this -> a); 
    this -> variable("pencil::b" , this -> b);
    this -> variable("pencil::c" , this -> c);
    this -> variable("pencil::d" , this -> d);
    this -> variable("pencil::e" , this -> e);
}

void Conuix::Sx_t::print(int p){
    this -> prec = p; 
    this -> variable("Sx::a" , this -> a); 
    this -> variable("Sx::b" , this -> b);
    this -> variable("Sx::c" , this -> c);
}

void Conuix::Sy_t::print(int p){
    this -> prec = p; 
    this -> variable("Sy::a" , this -> a); 
    this -> variable("Sy::b" , this -> b);
    this -> variable("Sy::c" , this -> c);
}


void Conuix::H_matrix_t::print(int p){
    this -> prec = p;

    std::cout << "HBAR" << std::endl;
    this -> HBX.print(p); 
    this -> HBS.print(p);
    this -> HBC.print(p);  

    std::cout << "H" << std::endl;
    this -> HTX.print(p); 
    this -> HTS.print(p);
    this -> HTC.print(p);  
}

long double Conuix::cos_theta(particle_template* jet, particle_template* lep){
    long double d12 = 0;
    d12 += (long double)(jet -> px) * (long double)(lep -> px); 
    d12 += (long double)(jet -> py) * (long double)(lep -> py); 
    d12 += (long double)(jet -> pz) * (long double)(lep -> pz); 
    return d12 / ((long double)(jet -> P) * (long double)(lep -> P)); 
}

long double Conuix::pencil_t::Z2(long double Sx, long double Sy){
    long double z_a = this -> a * Sx * Sx;
    long double z_b = this -> b * Sx * Sy;
    long double z_c = this -> c * Sy * Sy;
    long double z_d = this -> d * Sx + this -> e; 
    return z_a + z_b + z_c + z_d;
}

matrix_t Conuix::H_matrix_t::H_Matrix(long double tau, long double Z){
    return (this -> HBS * std::sinh(tau) + this -> HBC * std::cosh(tau) + this -> HBX) * std::fabs(Z); 
}

matrix_t Conuix::H_matrix_t::H_Tilde(long double tau, long double Z){
    return (this -> HTS * std::sinh(tau) + this -> HTC * std::cosh(tau) + this -> HTX) * std::fabs(Z); 
}

long double Conuix::Sx_t::Sx(long double tau, long double Z){
    return Z * (this -> a * std::cosh(tau) + this -> b * std::sinh(tau)) + this -> c; 
}

long double Conuix::Sy_t::Sy(long double tau, long double Z){
    return Z * (this -> a * std::cosh(tau) + this -> b * std::sinh(tau)) + this -> c; 
}

bool atomics_t::GetTauZ(long double sx, long double sy, long double* z, long double* t){
    long double f = this -> base.o / this -> base.beta; 

    long double kx = (this -> base.mass * this -> base.mass) / (this -> base.E * this -> base.beta);
    long double ky = (this -> base.tpsi * this -> base.E) / this -> base.beta;

    long double a = f * ( (sy + ky) * this -> base.cpsi - (sx + kx) * this -> base.spsi); 
    long double b =       (sx + kx) * this -> base.cpsi + (sy + ky) * this -> base.spsi; 

    if (std::fabs(a / b) >= 1.0){return false;}
    *t = std::atanh(a / b);
    *z = ( (this -> base.o * this -> base.cpsi) / this -> base.beta ) * std::cosh(*t);
    *z = (sx + kx) / (*z - this -> base.spsi * std::sinh(*t)); 
    return true; 
}

long double atomics_t::x1(long double Z, long double t){
    long double a = this -> base.beta * std::cosh(t) + this -> base.o * this -> base.tpsi * std::sinh(t); 
    return this -> base.beta * this -> base.E + std::fabs(Z) * (this -> base.cpsi / this -> base.o) * a; 
}

long double atomics_t::y1(long double Z, long double t){
    long double a = this -> base.beta * this -> base.tpsi * std::cosh(t) - this -> base.o * std::sinh(t); 
    return std::fabs(Z) * (this -> base.cpsi / this -> base.o) * a; 
}

void atomics_t::masses(long double Z, long double t, std::complex<long double>* mw, std::complex<long double>* mt){
    long double sx_ = this -> Sx.Sx(t, Z); 
    long double sy_ = this -> Sy.Sy(t, Z); 

    long double m2l = this -> base.mass * this -> base.mass; 
    long double m2j = this -> jt.mass * this -> jt.mass; 
    std::complex<long double> mw_ =     - m2l - 2 * this -> base.E * this -> base.beta * sx_; 
    std::complex<long double> mt_ = mw_ + m2j - 2 * this -> jt.energy * this -> jt.beta * (sy_ * this -> base.sin + sx_ * this -> base.cos); 
    *mw = std::sqrt(mw_); *mt = std::sqrt(mt_); 
}

void atomics_t::debug_mode(particle_template* jet, particle_template* lep){
    //this -> base.print(16); 
    long double rsx = -9409.045728244519 ; 
    long double rsy = -131393.59217560544; 
    long double cz(0), ct(0);
    this -> GetTauZ(rsx, rsy, &cz, &ct);
    this -> variable("cz", cz);
    this -> variable("ct", ct); 
    
    this -> variable("Sx", this -> Sx.Sx(ct, cz)); 
    this -> variable("Sy", this -> Sy.Sy(ct, cz)); 

    this -> variable("x1", this -> x1(cz, ct)); 
    this -> variable("y1", this -> y1(cz, ct));

    std::complex<long double> mw_, mt_;
    this -> masses(cz, ct, &mw_, &mt_); 
    this -> variable("mw", mw_.real()); 
    this -> variable("mt", mt_.real());

    this -> variable("Z2", this -> pencil.Z2(this -> Sx.Sx(ct, cz), this -> Sy.Sy(ct, cz)));

    this -> H_Matrix.H_Matrix(ct, cz).print(); 
    this -> H_Matrix.H_Tilde(ct, cz).print(); 
    this -> Mobius.init(&this -> base); 






}



