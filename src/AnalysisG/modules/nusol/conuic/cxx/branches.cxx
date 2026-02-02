#include <templates/particle_template.h>
#include <conuic/variables.h>
#include <common/matrix.h>

long double  signs(int sign, long double  v1, long double  v2){return (sign > 0) ? v1 : v2;}
long double* signs(int sign, long double* v1, long double* v2){return (sign > 0) ? v1 : v2;}

long double omega(long double sign, kinematic_c* data){
    return (1 / data -> sth) * (sign * data -> b_mu / data -> b_b - data -> cth); 
}

long double Omega(int sign, kinematic_c* data){
    long double w = signs(sign, data -> wp, data -> wm); 
    return w * w + 1 - data -> b_mu * data -> b_mu; 
}

long double pencil(int sign, kinematic_c* data){
    long double* a = signs(sign, &data -> z2p_a, &data -> z2m_a); 
    long double* b = signs(sign, &data -> z2p_b, &data -> z2m_b); 
    long double* c = signs(sign, &data -> z2p_c, &data -> z2m_c); 
    long double* d = signs(sign, &data -> z2p_d, &data -> z2m_d); 
    long double* e = signs(sign, &data -> z2p_e, &data -> z2m_e); 

    // --------- here ----------//





}


long double angles(particle_template* jet, particle_template* lep, kinematic_c* data){
    long double jx = (long double)jet -> px; long double lx = (long double)lep -> px;
    long double jy = (long double)jet -> py; long double ly = (long double)lep -> py;
    long double jz = (long double)jet -> pz; long double lz = (long double)lep -> pz;
    
    long double v1 = jx * jx + jy * jy + jz * jz; 
    long double v2 = lx * lx + ly * ly + lz * lz; 
    long double v3 = jx * lx + jy * ly + jz * lz; 

    long double lpe = (long double)(lep -> e);
    long double lpb = (long double)(lep -> beta); 
    data -> m_mu = std::sqrt(lpe * lpe - v2);
    data -> p_mu = lpb * lpe; 
    data -> b_mu = lpb; 

    long double bpe = (long double)(jet -> e);
    long double bpb = (long double)(jet -> beta); 
    data -> m_b = std::sqrt(bpe * bpe - v1);
    data -> p_b = bpe * bpb; 
    data -> b_b = bpb; 

    matrix_t vec = matrix_t(3, 1); 
    vec.at(0, 0) = jx; vec.at(1, 0) = jy; vec.at(2, 0) = jz; 

    long double phi   = std::atan2(ly, lx); 
    long double theta = std::acos(lz / (lpe * lpb));  
    matrix_t Rz(3, 3);
    Rz.at(0, 0) = std::cos(-phi); Rz.at(0, 1) = -std::sin(-phi); 
    Rz.at(1, 0) = std::sin(-phi); Rz.at(1, 1) =  std::cos(-phi);
    Rz.at(2, 2) = 1;
    
    matrix_t Ry(3, 3); 
    Ry.at(0, 0) =  std::cos(0.5 * M_PI - theta); Ry.at(0, 2) = std::sin(0.5 * M_PI - theta); 
    Ry.at(2, 0) = -std::sin(0.5 * M_PI - theta); Ry.at(2, 2) = std::cos(0.5 * M_PI - theta);
    Ry.at(1, 1) = 1;

    matrix_t b_p = Ry.dot(Rz.dot(vec)); 
    long double alpha = -std::atan2(b_p.at(2, 0), b_p.at(1, 0));

    matrix_t Rx(3, 3);
    Rx.at(0, 0) = 1; 
    Rx.at(1, 1) =  std::cos(alpha); Rx.at(1, 2) = -std::sin(alpha);
    Rx.at(2, 1) =  std::sin(alpha); Rx.at(2, 2) =  std::cos(alpha);
    data -> rot = new matrix_t(Rz.T().dot(Ry.T().dot(Rx.T()))); 
    return v3 / std::sqrt(v1 * v2); 
}




