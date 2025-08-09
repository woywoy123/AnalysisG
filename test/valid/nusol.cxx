#include "nusol.h"
#include <math.h>
#include <fstream>

double geo_t::px(double u, double v){return this -> center.x + this -> perp1.x*u + this -> perp2.x*v;}
double geo_t::py(double u, double v){return this -> center.y + this -> perp1.y*u + this -> perp2.y*v;}
double geo_t::pz(double u, double v){return this -> center.z + this -> perp1.z*u + this -> perp2.z*v;}

double geo_t::lx(double s){return this -> r0.x + this -> d.x*s;}
double geo_t::ly(double s){return this -> r0.y + this -> d.y*s;}
double geo_t::lz(double s){return this -> r0.z + this -> d.z*s;}


nusol_rev::nusol_rev(const particle& b, const particle& l){
    this -> bjet = new particle(b); 
    this -> lep  = new particle(l); 
    this -> para = new revised_t(); 

    double c  = cos_theta(this -> bjet, this -> lep); 
    double s  = std::pow(1.0 - c*c, 0.5);
    double w  = (this -> lep -> b / this -> bjet -> b - c)/s;
    double o2 = w*w + 1 - this -> lep -> b2; 
    this -> o = std::pow(o2, 0.5); 

    // coefficients
    this -> para -> A  = 1.0 / o2 - 1; 
    this -> para -> B  = - this -> lep -> m2 / (o2 * this -> lep -> e2); 
    this -> para -> C  = 2 * w / o2; 
    this -> para -> D  = 2 * this -> lep -> p; 
    this -> para -> F  = this -> lep -> m2; 

    // surface shifts 
    double wf = std::pow(1 + w*w, -0.5); 
    this -> para -> s0x = - this -> lep -> m2 / this -> lep -> p;
    this -> para -> s0y = - w * this -> lep -> p / this -> lep -> b2; 
    this -> para -> psi = std::atan(w); 
    this -> para -> cpsi = wf; 
    this -> para -> spsi = w * wf;
    this -> para -> lmb2 = this -> lep -> b2 / o2; 

    // ............ parameterization constants ............. //
    // Sx(t, Z) = |Z| * (a_x * cosh(t) + b_x * sinh(t)) + c_x;
    // a_x: (o / b_mu) * cos(psi)
    // b_x: - sin(psi)
    // c_x: - m^2_mu / p_mu
    this -> para -> a_x =  (this -> o / this -> lep -> b) * this -> para -> cpsi; 
    this -> para -> b_x = -this -> para -> spsi; 
    this -> para -> c_x =  this -> para -> s0x; 

    // Sy(t, Z) = |Z| * (a_y * cosh(t) + b_y * sinh(t)) + c_y;
    // a_y: (o / b_mu) * sin(psi)
    // b_y:  cos(psi)
    // c_y: - w * E^2_mu / p_mu
    this -> para -> a_y = (this -> o / this -> lep -> b) * this -> para -> spsi; 
    this -> para -> b_y = this -> para -> cpsi;
    this -> para -> c_y = this -> para -> s0y; 

    // mW^2: - m^2_mu - 2 * p_mu * Sx(t, Z)
    // mW^2: a_w + b_w * Sx(t, Z)
    this -> para -> a_w = - this -> lep -> m2; 
    this -> para -> b_w = - 2 * this -> lep -> p; 

    // mT^2: m^2_b - m^2_mu - 2 * (p_mu + p_b * cos(theta)) * Sx(t, Z) - 2 * p_b * sin(theta) * Sy(t, Z) 
    // mT^2: a_t + b_t * Sx(t, Z) + c_t * Sy(t, Z)
    this -> para -> a_t = this -> bjet -> m2 - this -> lep -> m2; 
    this -> para -> b_t = -2 * (this -> lep -> p + this -> bjet -> p * c); 
    this -> para -> c_t = -2 * this -> bjet -> b * s; 
    this -> make_rt(); 

    // ......... h_tilde definition: Z*(h1 + h2 * cosh(t) + h3 * sinh(t)) ........ //
    this -> para -> ht1.at(0,0) = 1.0 / this -> o; 
    this -> para -> ht1.at(1,0) = w   / this -> o; 
    this -> para -> ht1.at(2,1) = 1.0;

    this -> para -> ht2.at(0,2) = -this -> lep -> b * wf / this -> o;
    this -> para -> ht2.at(1,2) = -this -> lep -> b * wf * w/ this -> o;
    
    this -> para -> ht3.at(0,2) = -w * wf;
    this -> para -> ht3.at(1,2) =  wf;

    // ......... H definition: Z*(RT * h1 + RT * h2 * cosh(t) + RT * h3 * sinh(t)) ........ //
    this -> para -> hx1 = (*this -> rt) * this -> para -> ht1;
    this -> para -> hx2 = (*this -> rt) * this -> para -> ht2;
    this -> para -> hx3 = (*this -> rt) * this -> para -> ht3;
}

nusol_rev::~nusol_rev(){
    safe(&this -> bjet);
    safe(&this -> lep );
    safe(&this -> rt  ); 
    safe(&this -> para); 
}

void nusol_rev::make_rt(){
    if (this -> rt){return;}
    double phi_mu   = std::atan2(this -> lep -> py, this -> lep -> px);
    double theta_mu = std::acos( this -> lep -> pz / this -> lep -> p);

    matrix Rz(3,3); 
    Rz.at(0,0) =  std::cos(phi_mu); 
    Rz.at(0,1) = -std::sin(phi_mu); 
    Rz.at(2,2) = 1;
    Rz.at(1,0) = std::sin(phi_mu);
    Rz.at(1,1) = std::cos(phi_mu);
    
    matrix Ry(3,3); 
    Ry.at(0,0) = std::cos(theta_mu); 
    Ry.at(0,2) = std::sin(theta_mu); 
    Ry.at(1,1) = 1;
    Ry.at(2,0) = -std::sin(theta_mu); 
    Ry.at(2,2) =  std::cos(theta_mu);

    vec3 b_p = Ry * (Rz * vec3{this -> bjet -> px, this -> bjet -> py, this -> bjet -> pz});
    double alpha = -std::atan2(b_p.z, b_p.y);

    matrix Rx(3,3);
    Rx.at(0,0) = 1; 
    Rx.at(1,1) =  std::cos(alpha); 
    Rx.at(1,2) = -std::sin(alpha);
    Rx.at(2,1) =  std::sin(alpha); 
    Rx.at(2,2) =  std::cos(alpha);
    this -> rt = new matrix(Rz.T() * Ry.T() * Rx.T()); 
}

rev_t nusol_rev::translate(double sx, double sy){
    double z = 0; 
    z += this -> para -> A * sx*sx + this -> para -> B * sy*sy; 
    z += this -> para -> C * sx*sy + this -> para -> D * sx + this -> para -> F;
    z = std::pow(z, 0.5);

    rev_t out;
    // t: tanh^-1(o * (Sy - w*Sx + w*E_mu) / (b_mu * (Sx + w * Sy) + o^2 * E_mu))
    // 1. Translate the reference point to the hyperbola's center
    double u = sx - this -> para -> s0x;
    double v = sy - this -> para -> s0y; 

    // 2. Perform an inverse rotation to align with the hyperbola's principal axes
    double u_p =  u * this -> para -> cpsi + v * this -> para -> spsi;
    double v_p = -u * this -> para -> spsi + v * this -> para -> cpsi;

    // 3. Calculate the ratio a/b from the parameterization definitions.
    // a = |Z|*Omega/beta_mu and b = |Z|, so a/b = Omega/beta_mu
    double r = this -> o / this -> lep -> b * (v_p / u_p);

    // 4. Calculate t. The sign of v_prime determines the sign of t.
    double t = 0; 
    if (u_p > 0 && fabs(r) < 1){t  = std::atanh(r);}
    else if (u_p < 0 && v_p < 0){t = std::asinh(v_p / z);}
    else {t = std::acosh(fabs(u_p) * this -> lep -> b / (z * this -> o))*(1 - 2 * (v_p < 0));}
    out.t = t;  out.z = z; 
    out.v_p = v_p; out.u_p = u_p; 
    out.v   = v;   out.u   = u; 
    return out; 
}

mass_t nusol_rev::masses(double t, double z){
    mass_t out; 
    double sh = std::sinh(t); 
    double ch = std::cosh(t); 

    // Sx(t, Z) = |Z| * (a_x * cosh(t) + b_x * sinh(t)) + c_x;
    // a_x: (o / b_mu) * cos(psi)
    // b_x: - sin(psi)
    // c_x: - m^2_mu / p_mu
    out.sx = z * (this -> para -> a_x * ch + this -> para -> b_x * sh) + this -> para -> c_x; 

    // Sy(t, Z) = |Z| * (a_y * cosh(t) + b_y * sinh(t)) + c_y;
    // a_y: (o / b_mu) * sin(psi)
    // b_y:  cos(psi)
    // c_y: - w * E^2_mu / p_mu
    out.sy = z * (this -> para -> a_y * ch + this -> para -> b_y * sh) + this -> para -> c_y; 

    // mW^2: - m^2_mu - 2 * p_mu * Sx(t, Z)
    // mW^2: a_w + b_w * Sx(t, Z)
    double dw = this -> para -> b_w * out.sx + this -> para -> a_w; 
    out.mw = std::pow(std::abs(dw), 0.5); 

    // mT^2: m^2_b - m^2_mu - 2 * (p_mu + p_b * cos(theta)) * Sx(t, Z) - 2 * p_b * sin(theta) * Sy(t, Z) 
    // mT^2: a_t + b_t * Sx(t, Z) + c_t * Sy(t, Z)
    double dt = this -> para -> b_t * out.sx + this -> para -> c_t * out.sy + this -> para -> a_t;
    out.mt = std::pow(std::abs(dt), 0.5);
    return out; 
}

matrix nusol_rev::h_tilde(double t, double z){
    return (this -> para -> ht1 + this -> para -> ht2 * std::cosh(t) + this -> para -> ht3 * std::sinh(t)) * z;
}

matrix nusol_rev::H(double t, double z) const{
    return (this -> para -> hx1 + this -> para -> hx2 * std::cosh(t) + this -> para -> hx3 * std::sinh(t)) * z;
}

vec3 nusol_rev::v(double t, double z, double phi){
    return this -> H(t, z) * vec3{std::cos(phi), std::sin(phi), 1.0};
}

vec3 nusol_rev::center(double t, double z) const{
    return this -> H(t, z) * vec3{0, 0, 1};
}

vec3 nusol_rev::normal(double t, double z) const{
    matrix h = this -> H(t, z);
    vec3 A = h * vec3{1, 0, 0};
    return A.cross(h * vec3{0, 1, 0}); 
}

geo_t nusol_rev::geometry(double t, double z){
    geo_t gx;
    gx.center = this -> center(t, z); 

    vec3 N = this -> normal(t, z); 
    N = N * (1.0 / N.mag()); 
    bool sw = (std::abs(N.x) > 1e-6) * (std::abs(N.y) > 1e-6); 
    vec3 p1  = vec3{-sw*N.y + !sw, sw*N.x, 0};
    gx.perp1 = p1 * (1.0 / p1.mag()); 
    vec3 p2  = N.cross(gx.perp1); 
    gx.perp2 = p2 * (1.0 / p2.mag()); 
    return gx;
}

geo_t nusol_rev::intersection(const nusol_rev* nu, double t1, double z1, double t2, double z2) const{
    vec3 n1 = this -> normal(t1, z1); 
    vec3 n2 =   nu -> normal(t2, z2); 
    double n1_ = n1.dot(n1); 
    double n2_ = n2.dot(n2);
    double nx_ = n1.dot(n2);

    double a1 = n1.dot(this -> center(t1, z1)); 
    double a2 = n2.dot(  nu -> center(t2, z2)); 
    double fn1 = a1 * n2_ - a2 * nx_; 
    double fn2 = a2 * n1_ - a1 * nx_; 
   
    geo_t mx;  
    mx.d   = n1.cross(n2); 
    mx.r0  = (n1 * fn1  + n2 * fn2) * (1.0/(n1_ * n2_ - nx_*nx_)); 
    mx.nu1 = this -> intersection(mx.r0, mx.d, t1, z1); 
    mx.nu2 =   nu -> intersection(mx.r0, mx.d, t2, z2); 

    vec3 dp = mx.nu1 -> _pts1 - mx.nu2 -> _pts1; 
    vec3 dm = mx.nu1 -> _pts2 - mx.nu2 -> _pts2; 
    mx._d1 = dp.dot(dp);
    mx._d2 = dm.dot(dm); 
    mx.asym = std::abs(mx._d1 - mx._d2)/(0.5 * (mx._d1 + mx._d2)); 
    mx._d1 = std::pow(dp.dot(dp), 0.5);
    mx._d2 = std::pow(dm.dot(dm), 0.5); 
    return mx;
}


geo_t* nusol_rev::intersection(const vec3& r0, const vec3& d, double t, double z) const{
    matrix h = this -> H(t, z);
    vec3 A = h * vec3{1, 0, 0};
    vec3 B = h * vec3{0, 1, 0};
    vec3 Dx = r0 - this -> center(t, z);
    double a = 1.0 / A.mag2(), b = 1.0 / B.mag2(); 

    double alpha = a * Dx.dot(A);
    double beta  = a *  d.dot(A);
    double delta = b *  d.dot(B); 
    double gamma = b * Dx.dot(B);

    double x1 =  beta * beta + delta * delta;
    double x2 = alpha * beta + gamma * delta; 
    double x3 = alpha * alpha + gamma * gamma - 1; 
    double dx = x2 * x2 - x1 * x3;

    geo_t* mx = new geo_t(); 
    mx -> valid = dx >= 0; 
    dx = std::pow(std::abs(dx), 0.5); 

    mx -> _s1 = (-x2 + dx) / x1; 
    mx -> _s2 = (-x2 - dx) / x1; 
    mx -> _pts1 = r0 + d * mx -> _s1; 
    mx -> _pts2 = r0 + d * mx -> _s2;  
    mx -> _p1 = std::fmod(std::atan2(gamma + delta * mx -> _s1, alpha + beta * mx -> _s1), 2 * M_PI);
    mx -> _p2 = std::fmod(std::atan2(gamma + delta * mx -> _s2, alpha + beta * mx -> _s2), 2 * M_PI);
    if (mx -> _p1 > 2 * M_PI){mx -> _p1 = M_PI + std::fabs(mx -> _p1 - M_PI);}
    if (mx -> _p2 > 2 * M_PI){mx -> _p2 = M_PI + std::fabs(mx -> _p2 - M_PI);}
    if (mx -> _p1 < 0){mx -> _p1 = 2 *M_PI + mx -> _p1;}
    if (mx -> _p2 < 0){mx -> _p2 = 2 *M_PI + mx -> _p2;}

    return mx; 
}


void nusol_rev::export_ellipse(std::string name, double t, double z){
    matrix h = this -> H(t, z); 
    std::ofstream file("data/H" + name + ".csv");
    file << h.at(0, 0) << "," << h.at(0, 1) << "," << h.at(0, 2) << "\n";
    file << h.at(1, 0) << "," << h.at(1, 1) << "," << h.at(1, 2) << "\n";
    file << h.at(2, 0) << "," << h.at(2, 1) << "," << h.at(2, 2) << "\n";
    file.close();
}

void nusol_rev::export_ellipse(std::string filename, double t, double z, int n_pts){
    std::ofstream file("data/E" + filename + ".csv");
    for (int i(0); i <= n_pts; ++i){
        double phi = 2 * M_PI * i / n_pts;
        vec3 p = this -> v(t, z, phi);
        file << p.x << "," << p.y << "," << p.z << "\n";
    }
    file.close();
}



