#include <conuic/angular.h>
#include <conuic/atomics.h>
#include <math.h>

angular_t::angular_t(long double kappa, angle_t agl){
    auto lamb = [this](long double agl, long double c, long double s, long double t) -> void {
        this -> cos = c; this -> sin = s; this -> tan = t; this -> agl = agl; 
    };

    if (agl == angle_t::undef){lamb(kappa, std::cos(kappa), std::sin(kappa), std::tan(kappa)); return;}
    if (agl == angle_t::cos){lamb(std::acos(kappa), kappa        , cs_sin(kappa), cs_tan(kappa)); return;}
    if (agl == angle_t::sin){lamb(std::asin(kappa), sn_cos(kappa), kappa        , sn_tan(kappa)); return;}
    if (agl == angle_t::tan){lamb(std::atan(kappa), tn_cos(kappa), tn_sin(kappa),         kappa); return;}
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

matrix_t* angles(particle_template* jet, particle_template* lep){
    long double jx = convert(jet -> px); long double lx = convert(lep -> px);
    long double jy = convert(jet -> py); long double ly = convert(lep -> py);
    long double jz = convert(jet -> pz); long double lz = convert(lep -> pz);
    
    matrix_t vec = matrix_t(3, 1); 
    vec.at(0, 0) = jx; vec.at(1, 0) = jy; vec.at(2, 0) = jz; 

    long double phi   = std::atan2(ly, lx); 
    long double theta = std::acos(lz / convert(lep -> P));  

    angular_t rz(-phi); 
    angular_t ry(0.5 * M_PI - theta); 

    matrix_t b_p = ry.Ry().dot(rz.Rz().dot(vec)); 
    long double alpha = -std::atan2(b_p.at(2, 0), b_p.at(1, 0));

    angular_t rx(alpha);
    return new matrix_t(rz.Rz().T().dot(ry.Ry().T().dot(rx.Rx().T()))); 
}


