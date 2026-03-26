#include <conuic/atomics.h>
#include <conuic/data.h>
#include <math.h>

kinematics_t::~kinematics_t(){
    this -> ptr_ = nullptr; 
}

kinematics_t::kinematics_t(particle_template* ptr){
    this -> px = convert(ptr -> px);
    this -> py = convert(ptr -> py);
    this -> pz = convert(ptr -> pz);

    this -> e  = convert(ptr -> e   ); 
    this -> m  = convert(ptr -> mass);
    this -> b  = convert(ptr -> beta);
    this -> p  = convert(ptr -> P); 
    this -> ptr_ = ptr; 
}

branches_t::branches_t(){}
branches_t::~branches_t(){
    flush(&this -> CC); 
    flush(&this -> SC);
    flush(&this -> SS); 
}

hyper_t::hyper_t(long double tau){
    this -> cosh = std::cosh(tau); 
    this -> sinh = std::sinh(tau);
    this -> tanh = std::tanh(tau);
}

points_t::points_t(long double sx, long double sy, long double sz){
    this -> sx = sx; 
    this -> sy = sy; 
    this -> sz = sz; 
}

dline_t::dline_t(kinematics_t* kl, delta_t* dt, branches_t* br, int eps){
    this -> dtp = dt -> dp; this -> dtm = dt -> dm; 

    long double eps_ = (long double)eps;  
    this -> lx0 = (dt -> dp * br -> w * std::pow(kl -> e, 2) - std::pow(kl -> m, 2)) / kl -> p;
    this -> lcx = eps_ * (br -> O / kl -> b) * (1 - dt -> dp * br -> w) * br -> cpsi; 
    this -> lsx = - (br -> w + dt -> dp) * br -> cpsi; 

    this -> ly0 = (dt -> dm * br -> w * std::pow(kl -> e, 2) - std::pow(kl -> m, 2)) / kl -> p;
    this -> lcy = eps_ * (br -> O / kl -> b) * (1 - dt -> dm * br -> w) * br -> cpsi; 
    this -> lsy = - (br -> w + dt -> dm) * br -> cpsi; 
    this -> lb = br -> O / kl -> b; 
}


long double dline_t::dx(long double m_nu, long double tau, long double phi){
    return this -> lx0 + m_nu * (this -> lcx * std::cosh(tau) + this -> lsx * std::sinh(tau) * std::cos(phi)); 
}

long double dline_t::dy(long double m_nu, long double tau, long double phi){
    return this -> ly0 + m_nu * (this -> lcy * std::cosh(tau) + this -> lsy * std::sinh(tau) * std::cos(phi)); 
}


long double dline_t::Sdx(long double m_nu, long double tau, long double phi){
    long double x = - this -> dtm * this -> dx(m_nu, tau, phi) + this -> dtp * this -> dy(m_nu, tau, phi); 
    return x / (this -> dtp - this -> dtm); 
}

long double dline_t::Sdy(long double m_nu, long double tau, long double phi){
    long double x = - this -> dx(m_nu, tau, phi) + this -> dy(m_nu, tau, phi); 
    return x / (this -> dtp - this -> dtm); 
}

long double dline_t::U(long double m_nu, long double tau, long double phi){
    return this -> lb * std::sqrt(std::fabs(this -> dtp - this -> dtm)) * this -> dx(m_nu, tau, phi); 
}

long double dline_t::V(long double m_nu, long double tau, long double phi){
    return std::sqrt(std::fabs(this -> dtp - this -> dtm)) * this -> dy(m_nu, tau, phi); 
}





cline_t::cline_t(branches_t* br, long double dt, double eps, long double dti){
    this -> center = br -> sx0 - dt * br -> sy0; 

    this -> alpha  = eps * (br -> O / br -> bl) * (br -> cpsi - dt * br -> spsi); 
    this -> alpha_ = eps * (br -> O / br -> bl) * (br -> cpsi - dti * br -> spsi); 
    this -> beta   = br -> spsi + dt  * br -> cpsi; 
    this -> beta_  = br -> spsi + dti * br -> cpsi; 

    this -> theta   = std::atan2(this -> beta , this -> alpha ); 
    this -> theta_  = std::atan2(this -> beta_, this -> alpha_); 

    this -> tn     = this -> beta / this -> alpha;
    this -> cn     = tn_cos(this -> tn);
    this -> sn     = tn_sin(this -> tn); 
    this -> r      = std::sqrt(std::pow(this -> alpha, 2) - std::pow(this -> beta, 2)); 

    // ------ special cases ----------- //
    this -> zero_dd = std::atanh(this -> beta / this -> alpha); 
    
    // ------ jacobian ------- //
    this -> Jdet  = - (br -> O / br -> bl) * (dt - dti);
    this -> Jdet *= (br -> spsi * br -> spsi - br -> cpsi * br -> cpsi); 
    this -> Jdet *= eps; 
}

long double cline_t::fx(long double m_nu, long double tau, long double phi){
    long double o = (std::cosh(tau) * this -> cn - std::sinh(tau) * std::cos(phi) * this -> sn) * m_nu; 
    return this -> r * o + this -> center; 
}

long double cline_t::DfxDphi(long double m_nu, long double tau, long double phi){
    return (std::cosh(tau) * this -> cn - std::sinh(tau) * std::sin(phi) * this -> sn) * m_nu; 
}

long double cline_t::DfxDtau(long double m_nu, long double tau, long double phi){
    return (std::sinh(tau) * this -> cn - std::cosh(tau) * std::cos(phi) * this -> sn) * m_nu; 
}

long double cline_t::JacoDet(long double m_nu, long double tau, long double phi){
    return this -> Jdet * m_nu * m_nu * std::sinh(tau) * std::sinh(tau) * std::sin(phi); 
}

matrix_t cline_t::Jacobian(long double m_nu, long double tau, long double phi){
    matrix_t Jx(3,3); 
    hyper_t hx(tau); angular_t ax(phi); 
    Jx.at(0,0) = (this -> alpha  * hx.sinh - this -> beta  * hx.cosh * ax.cos) * m_nu;
    Jx.at(1,0) = (this -> alpha_ * hx.sinh - this -> beta_ * hx.cosh * ax.cos) * m_nu; 

    Jx.at(0,1) = this -> beta  * hx.sinh * ax.sin * m_nu;
    Jx.at(1,1) = this -> beta_ * hx.sinh * ax.sin * m_nu; 
    return Jx;  
}

long double cline_t::tau_degenJc(long double phi){
    angular_t ax(phi);
    long double r_ = (this -> alpha * this -> beta_ - this -> alpha_ * this -> beta) * ax.sin; 
    if (r_ < 0){return -1;}
    long double d = (this -> alpha + this -> beta_ * ax.sin); d *= d; 
    d -= 4 * (this -> alpha * this -> beta_ - this -> alpha_ * this -> beta) * ax.sin; 
    
    r_ = std::sqrt(r_); 
    long double r1 = this -> beta * ax.cos * (this -> alpha + this -> beta_ * ax.sin + 2 * r_) / d;
    long double r2 = this -> beta * ax.cos * (this -> alpha + this -> beta_ * ax.sin - 2 * r_) / d;
    if (std::fabs(r1) >=1 && std::fabs(r2) >= 1){return -1;}
    return (std::fabs(r1) >= 1) ? r2 : r1; 
}


angular_t::angular_t(long double kappa){
    this -> cos = std::cos(kappa); 
    this -> sin = std::sin(kappa);
    this -> tan = std::tan(kappa);
}

matrix_t angular_t::Rz(){
    matrix_t rot = matrix_t(3, 3);
    rot.at(0, 0) = this -> cos; rot.at(0, 1) = -this -> sin; 
    rot.at(1, 0) = this -> sin; rot.at(1, 1) =  this -> cos;
    rot.at(2, 2) = 1;
    return rot; 
}

matrix_t angular_t::Ry(){
    matrix_t rot(3, 3); 
    rot.at(0, 0) =  this -> cos; rot.at(0, 2) = this -> sin; 
    rot.at(1, 1) = 1;
    rot.at(2, 0) = -this -> sin; rot.at(2, 2) = this -> cos;
    return rot; 
}

matrix_t angular_t::Rx(){
    matrix_t rot(3, 3); 
    rot.at(0, 0) = 1; 
    rot.at(1, 1) = this -> cos; rot.at(1, 2) = -this -> sin;
    rot.at(2, 1) = this -> sin; rot.at(2, 2) =  this -> cos;
    return rot; 
}

