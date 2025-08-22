#include <reconstruction/multisol.h>
#include <reconstruction/solvers.h>
#include <reconstruction/mtx.h>
#include <fstream>
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


geo_t::~geo_t(){
    if (this -> nu1){delete this -> nu1;}
    if (this -> nu2){delete this -> nu2;}
}

double geo_t::px(double u, double v){
    return this -> center.x + this -> perp1.x*u + this -> perp2.x*v;
}

double geo_t::py(double u, double v){
    return this -> center.y + this -> perp1.y*u + this -> perp2.y*v;
}

double geo_t::pz(double u, double v){
    return this -> center.z + this -> perp1.z*u + this -> perp2.z*v;
}

double geo_t::lx(double s){
    return this -> r0.x + this -> d.x*s;
}

double geo_t::ly(double s){
    return this -> r0.y + this -> d.y*s;
}

double geo_t::lz(double s){
    return this -> r0.z + this -> d.z*s;
}


multisol::multisol(particle_template* bjet, particle_template* lep){
    this -> b_lep = _beta(lep); 
    this -> b_jet = _beta(bjet); 

    this -> p_lep = _mag(lep); 
    this -> p_jet = _mag(bjet); 

    this -> m2_lep = std::pow(lep -> mass, 2); 
    this -> m2_jet = std::pow(bjet -> mass, 2);  
    this -> e2_lep = std::pow(lep -> e, 2); 

    this -> phi_mu   = std::atan2(lep -> py, lep -> px);
    this -> theta_mu = std::acos( double(lep -> pz) / this -> p_lep);
    this -> vx_jet   = vec3{bjet -> px, bjet -> py, bjet -> pz};

    this -> para = new multisol_t(); 

    double c  = cos_theta(bjet, lep); 
    double s  = std::pow(1.0 - c*c, 0.5);
    this -> w = (this -> b_lep / this -> b_jet - c)/s;
    double o2 = this -> w * this -> w + 1 - this -> b_lep * this -> b_lep; 

    this -> o = std::pow(o2, 0.5); 

    // coefficients
    this -> para -> A  = 1.0 / o2 - 1; 
    this -> para -> B  = - this -> m2_lep / (o2 * this -> e2_lep); 
    this -> para -> C  = 2 * this -> w / o2; 
    this -> para -> D  = 2 * this -> p_lep; 
    this -> para -> F  = this -> m2_lep; 

    // surface shifts 
    double wf = std::pow(1 + this -> w * this -> w, -0.5); 
    this -> para -> s0x = - this -> m2_lep / this -> p_lep;
    this -> para -> s0y = - this -> w * this -> p_lep / (this -> b_lep * this -> b_lep); 
    this -> para -> psi = std::atan(this -> w); 
    this -> para -> cpsi = wf; 
    this -> para -> spsi = this -> w * wf;
    this -> para -> lmb2 = (this -> b_lep * this -> b_lep) / o2; 

    // ............ parameterization constants ............. //
    // Sx(t, Z) = |Z| * (a_x * cosh(t) + b_x * sinh(t)) + c_x;
    // a_x: (o / b_mu) * cos(psi)
    // b_x: - sin(psi)
    // c_x: - m^2_mu / p_mu
    this -> para -> a_x =  (this -> o / this -> b_lep) * this -> para -> cpsi; 
    this -> para -> b_x = -this -> para -> spsi; 
    this -> para -> c_x =  this -> para -> s0x; 

    // Sy(t, Z) = |Z| * (a_y * cosh(t) + b_y * sinh(t)) + c_y;
    // a_y: (o / b_mu) * sin(psi)
    // b_y:  cos(psi)
    // c_y: - w * E^2_mu / p_mu
    this -> para -> a_y = (this -> o / this -> b_lep) * this -> para -> spsi; 
    this -> para -> b_y = this -> para -> cpsi;
    this -> para -> c_y = this -> para -> s0y; 

    // mW^2: - m^2_mu - 2 * p_mu * Sx(t, Z)
    // mW^2: a_w + b_w * Sx(t, Z)
    this -> para -> a_w = - this -> m2_lep; 
    this -> para -> b_w = - 2 * this -> p_lep; 

    // mT^2: m^2_b - m^2_mu - 2 * (p_mu + p_b * cos(theta)) * Sx(t, Z) - 2 * p_b * sin(theta) * Sy(t, Z) 
    // mT^2: a_t + b_t * Sx(t, Z) + c_t * Sy(t, Z)
    this -> para -> a_t = this -> m2_jet - this -> m2_lep; 
    this -> para -> b_t = -2 * (this -> p_lep + this -> p_jet * c); 
    this -> para -> c_t = -2 * this -> b_jet * s; 
    
    this -> make_rt(); 

    // ......... h_tilde definition: Z*(h1 + h2 * cosh(t) + h3 * sinh(t)) ........ //
    this -> para -> ht1.at(0,0) = 1.0 / this -> o; 
    this -> para -> ht1.at(1,0) = this -> w / this -> o; 
    this -> para -> ht1.at(2,1) = 1.0;

    this -> para -> ht2.at(0,2) = -this -> b_lep * wf             / this -> o;
    this -> para -> ht2.at(1,2) = -this -> b_lep * wf * this -> w / this -> o;
    
    this -> para -> ht3.at(0,2) = -this -> w * wf;
    this -> para -> ht3.at(1,2) =  wf;

    // ......... H definition: Z*(RT * h1 + RT * h2 * cosh(t) + RT * h3 * sinh(t)) ........ //
    this -> para -> hx1 = (*this -> rt) * this -> para -> ht1;
    this -> para -> hx2 = (*this -> rt) * this -> para -> ht2;
    this -> para -> hx3 = (*this -> rt) * this -> para -> ht3;
}

multisol::~multisol(){
    safe(&this -> rt  ); 
    safe(&this -> para); 
}

void multisol::make_rt(){
    if (this -> rt){return;}

    matrix Rz(3,3); 
    Rz.at(0,0) =  std::cos(this -> phi_mu); 
    Rz.at(0,1) = -std::sin(this -> phi_mu); 
    Rz.at(2,2) = 1;
    Rz.at(1,0) = std::sin(this -> phi_mu);
    Rz.at(1,1) = std::cos(this -> phi_mu);
    
    matrix Ry(3,3); 
    Ry.at(0,0) = std::cos(this -> theta_mu); 
    Ry.at(0,2) = std::sin(this -> theta_mu); 
    Ry.at(1,1) = 1;
    Ry.at(2,0) = -std::sin(this -> theta_mu); 
    Ry.at(2,2) =  std::cos(this -> theta_mu);

    vec3 b_p = Ry * (Rz * this -> vx_jet); 
    double alpha = -std::atan2(b_p.z, b_p.y);

    matrix Rx(3,3);
    Rx.at(0,0) = 1; 
    Rx.at(1,1) =  std::cos(alpha); 
    Rx.at(1,2) = -std::sin(alpha);
    Rx.at(2,1) =  std::sin(alpha); 
    Rx.at(2,2) =  std::cos(alpha);
    this -> rt = new matrix(Rz.T() * Ry.T() * Rx.T()); 
}

rev_t multisol::translate(double sx, double sy){
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
    double r = this -> o / this -> b_lep * (v_p / u_p);

    // 4. Calculate t. The sign of v_prime determines the sign of t.
    double t = 0; 
    if (u_p > 0 && fabs(r) < 1){t  = std::atanh(r);}
    else if (u_p < 0 && v_p < 0){t = std::asinh(v_p / z);}
    else {t = std::acosh(fabs(u_p) * this -> b_lep / (z * this -> o))*(1 - 2 * (v_p < 0));}
    out.t = t;  out.z = z; 
    out.v_p = v_p; out.u_p = u_p; 
    out.v   = v;   out.u   = u; 
    return out; 
}

mass_t multisol::masses(double t, double z){
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

// ---------------- base matrices -------------- //
matrix multisol::H_tilde(double t, double z){
    return (this -> para -> ht1 + this -> para -> ht2 * std::cosh(t) + this -> para -> ht3 * std::sinh(t)) * z;
}

matrix multisol::dHdt_tilde(double t, double z){
    return (this -> para -> ht2 * std::sinh(t) + this -> para -> ht3 * std::cosh(t)) * z; 
}

matrix multisol::H(double t, double z){
    return (this -> para -> hx1 + this -> para -> hx2 * std::cosh(t) + this -> para -> hx3 * std::sinh(t)) * z;
}

matrix multisol::dHdt(double t, double z){
    return (this -> para -> hx2 * std::sinh(t) + this -> para -> hx3 * std::cosh(t)) * z;
}

matrix multisol::d2Hdt2(double t, double z){
    return (this -> para -> hx2 * std::cosh(t) + this -> para -> hx3 * std::sinh(t)) * z;
}

vec3 multisol::v(double t, double z, double phi){
    return this -> H(t, z) * vec3{std::cos(phi), std::sin(phi), 1.0};
}

vec3 multisol::dv_dt(double t, double z, double phi){
    return this -> dHdt(t, z) * vec3{std::cos(phi), std::sin(phi), 1.0};
}

vec3 multisol::dv_dphi(double t, double z, double phi){
    return this -> H(t, z) * vec3{-std::sin(phi), std::cos(phi), 1.0};
}

vec3 multisol::d2v_dt_dphi(double t, double z, double phi){
    return this -> dHdt(t, z) * vec3{-std::sin(phi), std::cos(phi), 1.0};
}

bool multisol::eigenvalues(double t, double z, vec3* real, vec3* imag){
    double r  = std::pow(1 + this -> w * this -> w, 0.5); 

    double a = 1.0; 
    double b = -z / this -> o; 
    double c = (z*z/(this -> o * r)) * (this -> b_lep * this -> w * std::cosh(t) - this -> o * std::sinh(t)); 
    double d = (z*z*z/this -> o) * r * std::sinh(t); 
    mtx* solx = solve_cubic(a, b, c, d);

    double vio = 0; 
    for (int x(0); x < 3; ++x){vio += std::fabs(solx -> _m[1][x]);}
    if (!real || !imag){delete solx; return vio > 10e-4;}

    real -> x = solx -> _m[0][0]; 
    real -> y = solx -> _m[0][1]; 
    real -> z = solx -> _m[0][2]; 
    
    imag -> x = solx -> _m[1][0]; 
    imag -> y = solx -> _m[1][1]; 
    imag -> z = solx -> _m[1][2]; 
    delete solx; 
    return vio > 10e-4; 
}

double multisol::dp_dt(){
    double r = std::pow(1 + this -> w * this -> w, 0.5); 
    double s = std::pow(this -> m2_lep / this -> e2_lep, 0.5); 

    double a = 1; 
    double b = -2 * this -> o; 
    double c = r * r * s; 
    double d =  this -> b_lep * this -> w * std::pow(r, 3);
    double e = -this -> b_lep * this -> w * this -> o * std::pow(r, 5); 
    mtx* slx = find_roots(a, b, c, d, e); 
    double t = std::atanh((this -> o - slx -> _m[0][0])/(this -> b_lep * this -> w)); 
//    std::cout << std::acosh(this -> w * std::pow(r, 3) / (this -> b_lep * this -> w)) << std::endl;  
    delete slx;
    return t; 
}


// ------------------ Properties ----------------------------- //
vec3 multisol::center(double t, double z){
    return this -> H(t, z) * vec3{0, 0, 1};
}

vec3 multisol::normal(double t, double z){
    matrix h = this -> H(t, z);
    vec3 A = h * vec3{1, 0, 0};
    return A.cross(h * vec3{0, 1, 0}); 
}


geo_t multisol::geometry(double t, double z){
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

geo_t multisol::intersection(multisol* nu, double t1, double z1, double t2, double z2){
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
    mx.valid = (mx.nu1 -> valid * mx.nu2 -> valid); 
    return mx;
}


geo_t* multisol::intersection(const vec3& r0, const vec3& d, double t, double z){
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


void multisol::export_ellipse(std::string name, double t, double z){
    matrix h = this -> H(t, z); 
    std::ofstream file("data/H" + name + ".csv");
    file << h.at(0, 0) << "," << h.at(0, 1) << "," << h.at(0, 2) << "\n";
    file << h.at(1, 0) << "," << h.at(1, 1) << "," << h.at(1, 2) << "\n";
    file << h.at(2, 0) << "," << h.at(2, 1) << "," << h.at(2, 2) << "\n";
    file.close();
}

void multisol::export_ellipse(std::string filename, double t, double z, int n_pts){
    std::ofstream file("data/E" + filename + ".csv");
    for (int i(0); i <= n_pts; ++i){
        double phi = 2 * M_PI * i / n_pts;
        vec3 p = this -> v(t, z, phi);
        file << p.x << "," << p.y << "," << p.z << "\n";
    }
    file.close();
}


// ---------- needs review ---------- //
//geo_t* multisol::intersection(const vec3& r0, const vec3& d, double t, double z){
//    // Fetch the ellipse's geometric properties (center and non-orthogonal axis vectors)
//    matrix h = this->H(t, z);
//    vec3 A = h * vec3{1, 0, 0};
//    vec3 B = h * vec3{0, 1, 0};
//    vec3 C = this->center(t, z);
//
//    // --- Step 1: Create an Orthonormal Basis for the Ellipse's Plane ---
//    // Use the Gram-Schmidt process to create a proper 2D coordinate system (e1, e2) from A and B.
//    vec3 e1 = A * (1.0 / A.mag());
//    vec3 B_perp = B - e1 * B.dot(e1);
//    vec3 e2 = B_perp * (1.0 / B_perp.mag());
//
//    // --- Step 2: Derive the Ellipse's Implicit Equation Coefficients ---
//    // These 'g' coefficients define the ellipse's shape in the (e1, e2) basis,
//    // correctly accounting for the skew from the non-orthogonal A and B vectors.
//    double A_mag2 = A.mag2();
//    double B_mag2 = B.mag2();
//    double AdotB = A.dot(B);
//    double det_inv = 1.0 / (A_mag2 * B_mag2 - AdotB * AdotB);
//
//    double g11 = B_mag2 * det_inv;
//    double g22 = A_mag2 * det_inv;
//    double g12 = -AdotB * det_inv;
//
//    // --- Step 3: Project the Line into the 2D Coordinate System ---
//    // Express the line's position in the (e1, e2) basis as a linear function of 's'.
//    vec3 Dx = r0 - C;
//    double x_prime_0 = Dx.dot(e1);
//    double y_prime_0 = Dx.dot(e2);
//    double dx_prime_ds = d.dot(e1);
//    double dy_prime_ds = d.dot(e2);
//
//    // --- Step 4: Form and Solve the Final Quadratic Equation for 's' ---
//    // Substitute the line equations into the implicit ellipse equation to get a
//    // quadratic equation of the form: c2*s^2 + c1*s + c0 = 0.
//    double c2 = g11*dx_prime_ds*dx_prime_ds + g22*dy_prime_ds*dy_prime_ds + 2*g12*dx_prime_ds*dy_prime_ds;
//    double c1 = 2 * (g11*x_prime_0*dx_prime_ds + g22*y_prime_0*dy_prime_ds + g12*(x_prime_0*dy_prime_ds + y_prime_0*dx_prime_ds));
//    double c0 = g11*x_prime_0*x_prime_0 + g22*y_prime_0*y_prime_0 + 2*g12*x_prime_0*y_prime_0 - 1;
//
//    // Calculate the discriminant to find the number of real solutions.
//    double discriminant = c1 * c1 - 4 * c2 * c0;
//
//    geo_t* mx = new geo_t();
//    if (discriminant < 0){
//        mx->valid = false; // No real intersection.
//        return mx;
//    }
//    mx->valid = true;
//
//    // Solve for the two values of s, which are the intersection points.
//    double sqrt_disc = std::sqrt(discriminant);
//    double two_c2_inv = 1.0 / (2 * c2);
//    mx->_s1 = (-c1 + sqrt_disc) * two_c2_inv;
//    mx->_s2 = (-c1 - sqrt_disc) * two_c2_inv;
//
//    // Calculate the 3D coordinates of the intersection points.
//    mx->_pts1 = r0 + d * mx->_s1;
//    mx->_pts2 = r0 + d * mx->_s2;
//
//    // --- Optional: Calculate the ellipse phase angle (phi) for each intersection ---
//    // This requires converting the (x', y') intersection coords back to the original (u, v) parameters.
//    double A_mag = A.mag();
//    double B_dot_e1 = B.dot(e1);
//    double B_dot_e2 = B.dot(e2);
//
//    // For the first point (s1)
//    double x_prime_s1 = x_prime_0 + mx->_s1 * dx_prime_ds;
//    double y_prime_s1 = y_prime_0 + mx->_s1 * dy_prime_ds;
//    double v1 = y_prime_s1 / B_dot_e2;
//    double u1 = (x_prime_s1 - v1 * B_dot_e1) / A_mag;
//    mx->_p1 = std::fmod(std::atan2(v1, u1) + 2 * M_PI, 2 * M_PI); // Normalize angle to [0, 2pi)
//
//    // For the second point (s2)
//    double x_prime_s2 = x_prime_0 + mx->_s2 * dx_prime_ds;
//    double y_prime_s2 = y_prime_0 + mx->_s2 * dy_prime_ds;
//    double v2 = y_prime_s2 / B_dot_e2;
//    double u2 = (x_prime_s2 - v2 * B_dot_e1) / A_mag;
//    mx->_p2 = std::fmod(std::atan2(v2, u2) + 2 * M_PI, 2 * M_PI); // Normalize angle to [0, 2pi)
//
//    return mx;
//}
