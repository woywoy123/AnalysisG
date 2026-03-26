#include <templates/particle_template.h>
#include <conuic/variables.h>
#include <conuic/atomics.h>
#include <conuic/angular.h>
#include <common/matrix.h>
#include <math.h>

angular_t::angular_t(){}; 

angular_t::angular_t(long double phi_){
    this -> cos = std::cos(phi_); 
    this -> sin = std::sin(phi_); 
    this -> tan = std::tan(phi_); 
    this -> phi = phi_; 
}

angular_t::angular_t(long double phi_, bool is_cos, bool is_sin, bool is_tan){
    if (is_sin){
        this -> sin = phi_; 
        this -> cos = std::sqrt(1 - phi_ * phi_); 
        this -> tan = this -> sin/this -> cos; 
        this -> phi = std::asin(phi_);
        return; 
    }

    if (is_cos){
        this -> cos = phi_; 
        this -> sin = std::sqrt(1 - phi_ * phi_); 
        this -> tan = this -> sin/this -> cos; 
        this -> phi = std::acos(phi_);
        return; 
    }

    if (is_tan){
        this -> tan = phi_; 
        this -> cos = 1 / std::sqrt(1 + phi_*phi_); 
        this -> sin = std::sqrt(1 - this -> cos * this -> cos); 
        this -> phi = std::atan(phi_);
        return; 
    }
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

hyper_t::hyper_t(long double phi_){
    this -> cosh = std::cosh(phi_); 
    this -> sinh = std::sinh(phi_); 
    this -> tanh = std::tanh(phi_); 

    this -> tau = phi_; 
    this -> mat = matrix_t(2, 1); 
    this -> mat.at(0, 0) = this -> cosh; 
    this -> mat.at(1, 0) = this -> sinh; 
}

shift_t::shift_t(long double sx_, long double sy_){
    this -> sx = sx_; 
    this -> sy = sy_;
    this -> dim = 2;  
}

shift_t::shift_t(long double sx_, long double sy_, long double sz_){
    this -> sx = sx_; 
    this -> sy = sy_; 
    this -> sz = sz_; 
    this -> dim = 3; 
}

matrix_t shift_t::to_mat(int dim_){
    if (dim_ < 0){dim_ = this -> dim;}
    matrix_t mx = matrix_t(dim_, 1); 
    mx.at(0, 0) = this -> sx; 
    mx.at(1, 0) = this -> sy;
    if (dim_ > 2){mx.at(2, 0) = this -> sz;} 
    return mx;  
}

void shift_t::print(){
    debug_s("Sx", this -> sx); 
    debug_s("Sy", this -> sy); 
    debug_s("Sz", this -> sz); 
    std::cout << std::endl;
}

long double angles(particle_template* jet, particle_template* lep, kinematic_c* data){
    long double jx = (long double)jet -> px; long double lx = (long double)lep -> px;
    long double jy = (long double)jet -> py; long double ly = (long double)lep -> py;
    long double jz = (long double)jet -> pz; long double lz = (long double)lep -> pz;
    
    long double v1 = jx * jx + jy * jy + jz * jz; 
    long double v2 = lx * lx + ly * ly + lz * lz; 
    long double v3 = jx * lx + jy * ly + jz * lz; 

    data -> e_mu = (long double)lep -> e;
    data -> m_mu = (long double)lep -> mass;
    data -> p_mu = (long double)lep -> P; 
    data -> b_mu = (long double)lep -> beta; 
    
    data -> e_b = (long double)jet -> e;
    data -> m_b = (long double)jet -> mass;
    data -> p_b = (long double)jet -> P;   
    data -> b_b = (long double)jet -> beta;

    matrix_t vec = matrix_t(3, 1); 
    vec.at(0, 0) = jx; vec.at(1, 0) = jy; vec.at(2, 0) = jz; 

    long double phi   = std::atan2(ly, lx); 
    long double theta = std::acos(lz / data -> p_mu);  

    angular_t rz(-phi); 
    angular_t ry(0.5 * M_PI - theta); 

    matrix_t b_p = ry.Ry().dot(rz.Rz().dot(vec)); 
    long double alpha = -std::atan2(b_p.at(2, 0), b_p.at(1, 0));

    angular_t rx(alpha);
    data -> rot = new matrix_t(rz.Rz().T().dot(ry.Ry().T().dot(rx.Rx().T()))); 
    return v3 / std::sqrt(v1 * v2); 
}


