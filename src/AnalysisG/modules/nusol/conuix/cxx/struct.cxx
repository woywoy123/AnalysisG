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

    // ------------------ Hardcore stuff ----------------- //
    this -> P      = new Conuix::characteristic::P_t(&this -> base);
    this -> dPdtau = new Conuix::characteristic::dPdtau_t(&this -> base); 
    this -> debug_mode(jet, lep); 
}

atomics_t::~atomics_t(){
    if (this -> P     ){delete this -> P;}
    if (this -> dPdtau){delete this -> dPdtau;}
    this -> P      = nullptr; 
    this -> dPdtau = nullptr; 
}








void Conuix::debug::print(int p){}
void Conuix::debug::variable(std::string name, long double val){
    std::cout << std::fixed << std::setprecision(this -> prec);
    std::cout << name << ": " << val << " | "; // std::endl;
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
    return (this -> HBS * std::sinh(tau) + this -> HBC * std::cosh(tau) + this -> HBX) * Z; 
}

matrix_t Conuix::H_matrix_t::H_Tilde(long double tau, long double Z){
    return (this -> HTS * std::sinh(tau) + this -> HTC * std::cosh(tau) + this -> HTX) * Z; 
}

long double Conuix::Sx_t::Sx(long double tau, long double Z){
    return Z * (this -> a * std::cosh(tau) + this -> b * std::sinh(tau)) + this -> c; 
}

long double Conuix::Sy_t::Sy(long double tau, long double Z){
    return Z * (this -> a * std::cosh(tau) + this -> b * std::sinh(tau)) + this -> c; 
}

void atomics_t::debug_mode(particle_template* jet, particle_template* lep){
    std::cout << "____________" << std::endl;
    std::cout << std::string(jet -> hash) << " " << std::string(lep -> hash) << std::endl;
    bool tp1 = std::string(lep -> hash) == "0xd0d956fefa26c936" &&  std::string(jet -> hash) == "0x4588f002767086e7";
    bool tp2 = std::string(lep -> hash) == "0x1dbf9ed01fdef73b" &&  std::string(jet -> hash) == "0x5a4ac8ead1f7d952";
    if (tp1 || tp2){std::cout << ">>>>>>>>Truth<<<<<<<<<<<<<<" << std::endl;}

//    std::cout << " Lepton: " << " ";
//    this -> lp.print(12);
//
//    std::cout << " Jet: " << " ";
//    this -> jt.print(12); 
//   
//    std::cout << "Base: " << " ";
//    this -> base.print(12); 
//
//    std::cout << "Pencil" << std::endl;
//    this -> pencil.print(12);  
//
//    std::cout << "Sx" << std::endl;
//    this -> Sx.print(12);  
//
//    std::cout << "Sy" << std::endl;
//    this -> Sy.print(12);  
//
//    std::cout << "Rotation" << std::endl;
//    this -> rotation.print(12); 
//  
//    std::cout << "H Matrix" << std::endl; 
//    this -> H_Matrix.print(12); 

//    this -> P -> print(12);  
    this -> dPdtau -> test(this); 
}



//------------ leptons ------------- 
//hash of lepton: 0x1dbf9ed01fdef73b top-index: 1
//hash of lepton: 0xd0d956fefa26c936 top-index: 0
//------------- b jets ------------- 
//hash of b-jet: 0x4588f002767086e7 top-index:0
//hash of b-jet: 0x5a4ac8ead1f7d952 top-index:1
//-------- Detector Objects -------- 
//px: 18412.8844 py: -223525.5748 pz: -767.0036 e: 225301.0938 h: 0x4588f002767086e7
//px: 103767.8340 py: -31521.9344 pz: -231695.9935 e: 256240.7500 h: 0x5a4ac8ead1f7d952
//px: -157773.9579 py: 82979.5096 pz: 206333.1078 e: 273569.1250 h: 0x8dd2fcc6f043e036
//px: -134399.7306 py: 552902.0706 pz: 638017.4176 e: 855419.5625 h: 0xcb87195a250740c6
//px: 59209.3143 py: 60491.7570 pz: -71514.6675 e: 111374.4062 h: 0xd59fcede6127c927
//px: 43868.7245 py: 47284.6158 pz: 49964.6735 e: 82947.0312 h: 0xdd2b0c07017da9fb
//px: 52844.3361 py: -98767.7979 pz: -356644.1539 e: 373821.6875 h: 0x1dbf9ed01fdef73b
//px: 7427.1530 py: -67723.8623 pz: -56225.4027 e: 88334.4766 h: 0xd0d956fefa26c936


