#include <conuic/conuic.h>
#include <conuic/atomics.h>
#include <conuic/constants.h>

matrix_t* angles(particle_template* jet, particle_template* lep){
    long double jx = (long double)jet -> px; long double lx = (long double)lep -> px;
    long double jy = (long double)jet -> py; long double ly = (long double)lep -> py;
    long double jz = (long double)jet -> pz; long double lz = (long double)lep -> pz;
    
    matrix_t vec = matrix_t(3, 1); 
    vec.at(0, 0) = jx; vec.at(1, 0) = jy; vec.at(2, 0) = jz; 

    long double phi   = std::atan2(ly, lx); 
    long double theta = std::acos(lz / (long double) lep -> P);  

    angular_t rz(-phi); 
    angular_t ry(0.5 * M_PI - theta); 

    matrix_t b_p = ry.Ry().dot(rz.Rz().dot(vec)); 
    long double alpha = -std::atan2(b_p.at(2, 0), b_p.at(1, 0));

    angular_t rx(alpha);
    return new matrix_t(rz.Rz().T().dot(ry.Ry().T().dot(rx.Rx().T()))); 
}

conuic::conuic(particle_template* jet, particle_template* lepton){
    this -> lep_ = new kinematics_t(lepton); 
    this -> jet_ = new kinematics_t(jet); 
    this -> lep_ ->  RT = angles(jet, lepton); 

    this -> plus  = build_branches(this -> jet_, this -> lep_, +1); 
    this -> minus = build_branches(this -> jet_, this -> lep_, -1); 
    this -> delG  = build_deltas(this -> plus, this -> minus); 
    
    this -> w_x_dt_x_eps_x = build_clines(this -> plus, this -> delG, +1.0);  
    this -> w_x_dt_I_eps_x = build_clines(this -> plus, this -> delG, +1.0);  
    
    this -> w_x_dt_x_eps_I = build_clines(this -> plus, this -> delG, -1.0);  
    this -> w_x_dt_I_eps_I = build_clines(this -> plus, this -> delG, -1.0);  

    this -> w_I_dt_x_eps_x = build_clines(this -> minus, this -> delG, +1.0);  
    this -> w_I_dt_I_eps_x = build_clines(this -> minus, this -> delG, +1.0);  
    
    this -> w_I_dt_x_eps_I = build_clines(this -> minus, this -> delG, -1.0);  
    this -> w_I_dt_I_eps_I = build_clines(this -> minus, this -> delG, -1.0);  

    this -> dt_w_x_eps_x = new dline_t(this -> lep_, this -> delG, this -> plus, +1); 
    this -> dt_w_x_eps_I = new dline_t(this -> lep_, this -> delG, this -> plus, -1); 
    this -> dt_w_I_eps_x = new dline_t(this -> lep_, this -> delG, this -> minus, +1); 
    this -> dt_w_I_eps_I = new dline_t(this -> lep_, this -> delG, this -> minus, -1); 

    build_tilde(this -> plus , this -> lep_);
    build_tilde(this -> minus, this -> lep_); 
    this -> solve(); 
}

conuic::~conuic(){
    flush(&this -> lep_);
    flush(&this -> jet_);   
    flush(&this -> plus);
    flush(&this -> minus); 
    flush(&this -> delG); 
    flush(&this -> RT); 

    flush(&this -> w_x_dt_x_eps_x); 
    flush(&this -> w_x_dt_I_eps_x); 

    flush(&this -> w_x_dt_x_eps_I); 
    flush(&this -> w_x_dt_I_eps_I); 

    flush(&this -> w_I_dt_x_eps_x); 
    flush(&this -> w_I_dt_I_eps_x); 

    flush(&this -> w_I_dt_x_eps_I); 
    flush(&this -> w_I_dt_I_eps_I); 

    flush(&this -> dt_w_x_eps_x); 
    flush(&this -> dt_w_x_eps_I); 

    flush(&this -> dt_w_I_eps_x); 
    flush(&this -> dt_w_I_eps_I); 
}

branches_t* conuic::brn(int sign){return route(this -> plus, this -> minus, sign);}

long double conuic::Z2(long double sx, long double sy, long double m_nu, int sign){
    branches_t* br = route(this -> plus, this -> minus, sign); 
    long double z2 = br -> A * sx * sx;
    z2 += br -> B * sx * sy;
    z2 += br -> C * sy * sy;
    z2 += br -> D * sx;
    z2 += br -> E;
    return z2 - m_nu * m_nu; 
}

std::complex<long double> conuic::mW(long double sx, long double m_nu){
    return std::sqrt(std::complex<long double>(std::pow(m_nu, 2) - std::pow(this -> lep_ -> m, 2) - 2 * this -> lep_ -> p * sx)); 
}

std::complex<long double> conuic::mT(long double sx, long double sy, long double m_nu){
    long double m_ = std::pow(m_nu, 2) - std::pow(this -> lep_ -> m, 2) + std::pow(this -> jet_ -> m, 2); 
    long double x_ = 2 * (this -> lep_ -> p * sx + this -> jet_ -> p * (this -> minus -> cth * sx + this -> minus -> sth * sy));
    return std::sqrt(std::complex<long double>(m_ - x_)); 
}

long double conuic::Sx(long double tau, long double kappa, long double m_nu, int sign, int eps){
    hyper_t hx(tau); angular_t ax(kappa); 
    branches_t* br = this -> brn(sign); 
    long double bl = this -> lep_ -> b; 
    long double sh = convert(eps); 
    return (m_nu / bl) * (sh * br -> O * hx.cosh * br -> cpsi - bl  * hx.sinh * ax.cos * br -> spsi) + br -> sx0; 
}

long double conuic::Sy(long double tau, long double kappa, long double m_nu, int sign, int eps){
    hyper_t hy(tau); angular_t ay(kappa); 
    branches_t* br = this -> brn(sign); 
    long double bl = this -> lep_ -> b; 
    long double sh = convert(eps); 
    return (m_nu / bl) * (sh * br -> O * hy.cosh * br -> spsi + bl * hy.sinh * ay.cos * br -> cpsi) + br -> sy0; 
}

long double conuic::Z(long double tau, long double kappa, long double m_nu, int sign){
    hyper_t hz(tau); angular_t az(kappa); 
    return m_nu * hz.sinh * az.sin; 
}

points_t conuic::S(long double tau, long double kappa, long double m_nu, int sign, int eps){
    return points_t(
            this -> Sx(tau, kappa, m_nu, sign, eps), 
            this -> Sy(tau, kappa, m_nu, sign, eps), 
            this ->  Z(tau, kappa, m_nu, sign)
    ); 
}

long double conuic::x1(long double tau, long double kappa, long double m_nu, int sign, int eps){
    hyper_t hx(tau); angular_t ax(kappa); 
    branches_t* br = this -> brn(sign); 
    long double bl = this -> lep_ -> b; 
    long double O  = br -> O; 
    long double sh = convert(eps); 
    return - (m_nu / O) * (sh * bl * br -> cpsi * hx.cosh + O * br -> spsi * hx.sinh * ax.cos) + this -> lep_ -> p; 
}

long double conuic::y1(long double tau, long double kappa, long double m_nu, int sign, int eps){
    hyper_t hy(tau); angular_t ay(kappa); 
    branches_t* br = this -> brn(sign); 
    long double bl = this -> lep_ -> b; 
    long double O  = br -> O; 
    long double sh = convert(eps); 
    return (m_nu / O) * (O * br -> cpsi * hy.sinh * ay.cos - sh * bl * br -> spsi * hy.cosh); 
}

matrix_t conuic::H_tilde(long double tau, long double kappa, long double m_nu, int sign, int eps){
    hyper_t ht(tau); angular_t at(kappa); 
    branches_t* br = this -> brn(sign); 
    long double sh = convert(eps); 
    return m_nu * ( sh * (*br -> CC) * ht.cosh + (*br -> SC) * ht.sinh * at.cos + (*br -> SS) * ht.sinh * at.sin );  
}

long double conuic::line(long double sx, long double sy, int sign){
    long double* dl = route(&this -> delG -> dp, &this -> delG -> dm, sign);  
    return (sx - (*dl) * sy); 
}

long double conuic::dG2(long double sx, long double sy){
    return - this -> delG -> Gm * this -> delG -> Gp * this -> line(sx, sy, -1) * this -> line(sx, sy, +1);
}


void conuic::solve(){
    long double Op = this -> plus -> O; 
    long double wp = this -> plus -> tpsi;
    long double bl = this -> plus -> bl;  

    long double Om = this -> minus -> O;
    long double wm = this -> minus -> tpsi;

    std::cout << "==================================" << std::endl;
    long double tau1 = 0; 
    long double tau2 = 0; 

    // 1. -----
    //long double tau1 = std::fabs(Op*Op - Om*Om) / (2 * Op * Om * (1 - std::pow((wp - wm)/(wp + wm), 2))); 

    // 2. -----
    //tau1 = std::sqrt(1 + (1 + wp * wp)/(bl * bl) + (1 + wp*wp) * (1 + wp * wm)/(bl * bl * Op * Om));
    //tau2 = std::sqrt(1 + (1 + wp * wp)/(bl * bl) - (1 + wp*wp) * (1 + wp * wm)/(bl * bl * Op * Om));

    // 3. ----    
    tau1 = this -> w_x_dt_x_eps_I -> zero_dd; 
    tau2 = this -> w_x_dt_I_eps_x -> zero_dd; 
 
    std::cout << "______________________________" << std::endl;
    long double phi = 0; 
    std::cout << tools::to_string(tau1, 12) << std::endl;
    std::cout << tools::to_string(tau2, 12) << std::endl;

    std::cout << tools::to_string(m_nuG(this -> delG, this -> plus, this -> lep_, tau1, 0, +1, +1), 12) << std::endl; 
    std::cout << tools::to_string(m_nuG(this -> delG, this -> plus, this -> lep_, tau1, 0, -1, +1), 12) << std::endl; 
    std::cout << tools::to_string(m_nuG(this -> delG, this -> plus, this -> lep_, tau1, 0, +1, -1), 12) << std::endl; 
    std::cout << tools::to_string(m_nuG(this -> delG, this -> plus, this -> lep_, tau1, 0, -1, -1), 12) << std::endl; 


    std::cout << tools::to_string(m_nuG(this -> delG, this -> minus, this -> lep_, tau2, 0, +1, +1), 12) << std::endl; 
    std::cout << tools::to_string(m_nuG(this -> delG, this -> minus, this -> lep_, tau2, 0, -1, +1), 12) << std::endl; 
    std::cout << tools::to_string(m_nuG(this -> delG, this -> minus, this -> lep_, tau2, 0, +1, -1), 12) << std::endl; 
    std::cout << tools::to_string(m_nuG(this -> delG, this -> minus, this -> lep_, tau2, 0, -1, -1), 12) << std::endl; 












}








