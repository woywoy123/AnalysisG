#include <conics/nusol.h>
#include <math.h>

double _mag(particle_template* p){
    double p2 = double(p -> px) * double(p -> px); 
    p2 += double(p -> py) * double(p -> py); 
    p2 += double(p -> pz) * double(p -> pz); 
    return std::pow(p2, 0.5); 
}

double _beta(particle_template* prt){
    return _mag(prt) / double(prt -> e);
}

double cos_theta(particle_template* b, particle_template* mu){
    double _d = double(b -> px) * double(mu -> px);
    _d += double(b -> py) * double(mu -> py); 
    _d += double(b -> pz) * double(mu -> pz); 
    return _d / (_mag(b) * _mag(mu));
}

nuclx_t::nuclx_t(particle_template* b, particle_template* mu){

    // --------- fetch all the kinematics --------- //
    this -> mass_lep = mu -> mass; this -> mass_jet = b -> mass; 
    this -> beta_lep = _beta(mu);  this -> beta_jet = _beta(b); 
    this -> p_lep    = _mag(mu);   this -> p_jet    = _mag(b); 
    this -> e_lep    = mu -> e;    this -> e_jet    = b -> e; 
    this -> phi_mu   = std::atan2(mu -> py, mu -> px); 
    this -> theta_mu = std::acos(mu -> pz / this -> p_lep); 
    this -> cos_t    = cos_theta(b, mu); 
    this -> sin_t    = std::pow(1 - this -> cos_t * this -> cos_t, 0.5); 
    
    // ---------- define the main statics -------- //
    this -> w  = ((this -> beta_lep / this -> beta_jet) - this -> cos_t)/this -> sin_t; 
    this -> w2 = this -> w * this -> w; 
    this -> wr = std::pow(1 + this -> w2, -0.5); 

    this -> o2 = this -> w2 + std::pow(std::abs(this -> mass_lep / this -> e_lep), 2); 
    this -> o  = std::pow(this -> o2, 0.5);

    // ---------- build rotation --------- //
    this -> vec_jet = matrix_t(3, 1); 
    this -> vec_jet.at(0, 0) = b -> px; 
    this -> vec_jet.at(1, 0) = b -> py;
    this -> vec_jet.at(2, 0) = b -> pz; 
    this -> rotation(); 

    // ---------- derive coefficients ------------ //
    // Z^2 = A Sx^2 + B Sy^2 + C Sx*Sy + D * Sx + F
    this -> surface(); 

    // finds the center of the surface and shifts
    this -> shifts(); 

    // pre-define H_bar and H matrices
    this -> H_bar(); this -> H(); 

    // --------- derive the reverse mapping ------- //
    // this is used for computing the reverse mass parameterization
    this -> sx(); this -> sy(); 

    // --------- mass reverse mapping polynomial -------- //
    this -> mw(); this -> mt();  

    // --------- polnomial --------- //
    this -> polynomial(); 


}

void nuclx_t::polynomial(){
    // characteristic polynomial coefficients
    // a_l: - 1.0 / o 
    // b_l: beta_lep * w / (o * sqrt(1 + w*w))
    // c_l: - 1.0 / sqrt(1 + w*w)
    // d_l: sqrt(1 + w*w)/o;  
    this -> a_l = - 1.0 / this -> o; 
    this -> b_l = this -> beta_lep * this -> w * this -> wr / this -> o; 
    this -> c_l = - this -> wr; 
    this -> d_l = std::pow(1 + this -> w2, 0.5) / this -> o; 
}

void nuclx_t::critical_t0(){
    double r = std::pow(1 - this -> beta_lep * this -> beta_lep, 0.5); 
    double a = this -> beta_lep * this -> w * this -> wr * 1.0 / r; 
    double b = std::pow(1 + this -> w2, 0.5) / (3 * this -> o * r); 
    this -> t_0 = std::asinh(a) - std::asinh(b); 
}

void nuclx_t::rotation(){
    matrix_t Rz(3, 3); 
    Rz.at(0, 0) =  std::cos(this -> phi_mu); 
    Rz.at(0, 1) = -std::sin(this -> phi_mu); 
    Rz.at(2, 2) = 1;
    Rz.at(1, 0) = std::sin(this -> phi_mu);
    Rz.at(1, 1) = std::cos(this -> phi_mu);
    
    matrix_t Ry(3, 3); 
    Ry.at(0, 0) = std::cos(this -> theta_mu); 
    Ry.at(0, 2) = std::sin(this -> theta_mu); 
    Ry.at(1, 1) = 1;
    Ry.at(2, 0) = -std::sin(this -> theta_mu); 
    Ry.at(2, 2) =  std::cos(this -> theta_mu);

    matrix_t b_p = Ry.dot(Rz.dot(this -> vec_jet)); 
    double alpha = -std::atan2(b_p.at(2, 0), b_p.at(1, 0));

    matrix_t Rx(3, 3);
    Rx.at(0, 0) = 1; 
    Rx.at(1, 1) =  std::cos(alpha); 
    Rx.at(1, 2) = -std::sin(alpha);
    Rx.at(2, 1) =  std::sin(alpha); 
    Rx.at(2, 2) =  std::cos(alpha);
    this -> R_T  =  matrix_t(Rz.T().dot(Ry.T().dot(Rx.T()))); 
}

void nuclx_t::H_bar(){
    // H_bar = 1.0/O * [ HC + (beta_mu / sqrt(1 + w^2)) * H1 * cosh(t) + (O / sqrt(1 + w^2)) * H2 * sinh(t) ];
    // HC = [ [1, 0,  0], [w, 0,  0], [0, O, 0] ]
    // H1 = [ [0, 0, -1], [0, 0, -w], [0, 0, 0] ]
    // H2 = [ [0, 0, -w], [0, 0,  1], [0, 0, 0] ]
    this -> HBc.at(0, 0) = 1.0 / this -> o;
    this -> HBc.at(1, 0) = this -> w / this -> o; 
    this -> HBc.at(2, 1) = 1.0; 
    
    this -> HB1.at(0, 2) = - this -> beta_lep * this -> wr / this -> o; 
    this -> HB1.at(1, 2) = - this -> beta_lep * this -> wr * this -> w / this -> o; 
    
    this -> HB2.at(0, 2) = - this -> w * this -> wr; 
    this -> HB2.at(1, 2) =   this -> wr; 
}

void nuclx_t::H(){
    // same as H_Bar but with rotation: R_T
    this -> Hc = this -> R_T.dot(this -> HBc);
    this -> H1 = this -> R_T.dot(this -> HB1); 
    this -> H2 = this -> R_T.dot(this -> HB2); 
}

void nuclx_t::surface(){
    this -> A = 1.0 / this -> o2 - 1; 
    this -> B = - std::pow(this -> mass_lep / this -> e_lep, 2) * (1.0 / this -> o2); 
    this -> C = 2 * this -> w / this -> o2; 
    this -> D = 2 * this -> p_lep; 
    this -> F = this -> mass_lep * this -> mass_lep; 
}

void nuclx_t::shifts(){
    this -> s0x = - std::pow(this -> mass_lep, 2) / this -> p_lep;  
    this -> s0y = - this -> w * this -> p_lep / (this -> beta_lep * this -> beta_lep); 
    this -> psi = std::atan(this -> w); 

    // cos(psi) and sin(psi); 
    this -> cpsi = this -> wr; 
    this -> spsi = this -> w * this -> wr; 

    // only need to compute a single eigenvalue, the other one is -1
    this -> lmb2 = (this -> beta_lep * this -> beta_lep); 
}

void nuclx_t::sx(){
    // Sx(t, Z) = |Z| * (a_x * cosh(t) + b_x * sinh(t)) + c_x;
    // a_x: (o / b_mu) * cos(psi)
    // b_x: - sin(psi)
    // c_x: - m^2_mu / p_mu
    this -> a_x =  (this -> o / this -> beta_lep) * this -> cpsi; 
    this -> b_x = -this -> spsi; 
    this -> c_x =  this -> s0x; 
}

void nuclx_t::sy(){
    // Sy(t, Z) = |Z| * (a_y * cosh(t) + b_y * sinh(t)) + c_y;
    // a_y: (o / b_mu) * sin(psi)
    // b_y:  cos(psi)
    // c_y: - w * E^2_mu / p_mu
    this -> a_y = (this -> o / this -> beta_lep) * this -> spsi; 
    this -> b_y = this -> cpsi;
    this -> c_y = this -> s0y; 
}

void nuclx_t::mw(){
    // mW^2: - m^2_mu - 2 * p_mu * Sx(t, Z)
    // mW^2: a_w + b_w * Sx(t, Z)
    this -> a_w = - this -> mass_lep * this -> mass_lep; 
    this -> b_w = - 2 * this -> p_lep; 
}

void nuclx_t::mt(){
    // mT^2: m^2_b - m^2_mu - 2 * (p_mu + p_b * cos(theta)) * Sx(t, Z) - 2 * p_b * sin(theta) * Sy(t, Z) 
    // mT^2: a_t + b_t * Sx(t, Z) + c_t * Sy(t, Z)
    this -> a_t = this -> mass_jet * this -> mass_jet - this -> mass_lep * this -> mass_lep; 
    this -> b_t = -2 * (this -> p_lep + this -> p_jet * this -> cos_t); 
    this -> c_t = -2 * this -> beta_jet * this -> sin_t;  
}


nuclx_t nuclx_t::from_sx_sy(double _sx, double _sy){
    nuclx_t out = *this; 

    // t: tanh^-1(o * (Sy - w*Sx + w*E_mu) / (b_mu * (Sx + w * Sy) + o^2 * E_mu))
    // Translate the reference point to the hyperbola's center
    out.u = _sx - this -> s0x; 
    out.v = _sy - this -> s0y; 

    // Perform an inverse rotation to align with the hyperbola's principal axes
    out.u_p =  out.u * this -> cpsi + out.v * this -> spsi; 
    out.v_p = -out.u * this -> spsi + out.v * this -> cpsi; 
    out.z_v = this -> get_z(_sx, _sy); 
    out.t_v = this -> get_t(out.u_p, out.v_p, out.z_v); 
    return out; 
} 

double nuclx_t::get_z(double _sx, double _sy){
    double _z = this -> A * _sx * _sx + this -> B * _sy * _sy + this -> C * _sx * _sy + this -> D * _sx + this -> F;
    return (_z >= 0) ? std::pow(_z, 0.5) : -1.0; 
}

double nuclx_t::get_t(double _up, double _vp, double _z){
    // Calculate the ratio a/b from the parameterization definitions.
    // a = |Z|*Omega/beta_mu and b = |Z|, so a/b = Omega/beta_mu
    double r = this -> o / this -> beta_lep * (_vp / _up);

    // Calculate t. The sign of v_prime determines the sign of t.
    double _t = 0; 
    if (_up > 0 && std::fabs(r) < 1){_t = std::atanh(r);}
    else if (_up < 0 && _vp < 0){_t = std::asinh(_vp / _z);}
    else {_t = std::acosh(fabs(_up) * this -> beta_lep / (_z * this -> o))*(1 - 2 * (_vp < 0));}
    return _t; 
}

nuclx_t nuclx_t::from_z_t(double _z, double _t){
    nuclx_t out = *this; 
    double sh = std::sinh(_t); 
    double ch = std::cosh(_t); 

    // Sx(t, Z) = |Z| * (a_x * cosh(t) + b_x * sinh(t)) + c_x;
    // a_x: (o / b_mu) * cos(psi)
    // b_x: - sin(psi)
    // c_x: - m^2_mu / p_mu
    out.sx_v = _z * (this -> a_x * ch + this -> b_x * sh) + this -> c_x; 

    // Sy(t, Z) = |Z| * (a_y * cosh(t) + b_y * sinh(t)) + c_y;
    // a_y: (o / b_mu) * sin(psi)
    // b_y:  cos(psi)
    // c_y: - w * E^2_mu / p_mu
    out.sy_v = _z * (this -> a_y * ch + this -> b_y * sh) + this -> c_y; 
    
    out.mt_v = this -> get_mt(out.sx_v, out.sy_v); 
    out.mw_v = this -> get_mw(out.sx_v); 
    return out; 
} 

double nuclx_t::get_mt(double _sx, double _sy){
    // mT^2: m^2_b - m^2_mu - 2 * (p_mu + p_b * cos(theta)) * Sx(t, Z) - 2 * p_b * sin(theta) * Sy(t, Z) 
    // mT^2: a_t + b_t * Sx(t, Z) + c_t * Sy(t, Z)
    return std::pow(std::abs(this -> b_t * _sx + this -> c_t * _sy + this -> a_t), 0.5);
}

double nuclx_t::get_mw(double _sx){
    // mW^2: - m^2_mu - 2 * p_mu * Sx(t, Z)
    // mW^2: a_w + b_w * Sx(t, Z)
    return std::pow(std::abs(this -> b_w * _sx + this -> a_w), 0.5); 
}


