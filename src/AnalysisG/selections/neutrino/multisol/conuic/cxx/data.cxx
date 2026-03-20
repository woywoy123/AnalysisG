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





