#include <templates/particle_template.h>
#include <conuix/htilde.h>
#include <conuix/struct.h>
#include <cmath>

void Conuix::get_kinematic(particle_template* ptr, Conuix::kinematic_t* kin){
    kin -> beta = (long double)(ptr -> beta);
    kin -> mass = (long double)(std::abs(ptr -> mass)); 
    kin -> energy = (long double)(ptr -> e);
}

void Conuix::get_rotation(particle_template* jet, particle_template* lep, Conuix::rotation_t* rot){
    long double lpx = (long double)(lep -> px);
    long double lpy = (long double)(lep -> py);
    long double lpz = (long double)(lep -> pz);
    long double lpe = (long double)(lep -> e);
    long double lpb = (long double)(lep -> beta); 

    rot -> phi   = std::atan2(lpy, lpx); 
    rot -> theta = std::acos(lpz / (lpe * lpb));  

    rot -> vec = matrix_t(3, 1); 
    rot -> vec.at(0, 0) = (long double)(jet -> px); 
    rot -> vec.at(1, 0) = (long double)(jet -> py); 
    rot -> vec.at(2, 0) = (long double)(jet -> pz); 
    
    matrix_t Rz(3, 3);
    Rz.at(0, 0) =  std::cos(-rot -> phi); 
    Rz.at(0, 1) = -std::sin(-rot -> phi); 
    Rz.at(2, 2) = 1;
    Rz.at(1, 0) = std::sin(-rot -> phi);
    Rz.at(1, 1) = std::cos(-rot -> phi);
    
    matrix_t Ry(3, 3); 
    Ry.at(0, 0) = std::cos(0.5 * M_PI - rot -> theta); 
    Ry.at(0, 2) = std::sin(0.5 * M_PI - rot -> theta); 
    Ry.at(1, 1) = 1;
    Ry.at(2, 0) = -std::sin(0.5 * M_PI - rot -> theta); 
    Ry.at(2, 2) =  std::cos(0.5 * M_PI - rot -> theta);

    matrix_t b_p = Ry.dot(Rz.dot(rot -> vec)); 
    long double alpha = -std::atan2(b_p.at(2, 0), b_p.at(1, 0));

    matrix_t Rx(3, 3);
    Rx.at(0, 0) = 1; 
    Rx.at(1, 1) =  std::cos(alpha); 
    Rx.at(1, 2) = -std::sin(alpha);
    Rx.at(2, 1) =  std::sin(alpha); 
    Rx.at(2, 2) =  std::cos(alpha);
    rot -> R_T = matrix_t(Rz.T().dot(Ry.T().dot(Rx.T()))); 
}

void Conuix::get_psi_theta_mapping(Conuix::base_t* base, Conuix::thetapsi_t* msp){
    long double r = base -> rbl; 
    long double d = std::pow(1 + base -> w * base -> w - r * r, 0.5); 
    msp -> p_sin = (r * base -> w + d) / (1 + base -> w * base -> w); 
    msp -> m_sin = (r * base -> w - d) / (1 + base -> w * base -> w); 

    msp -> p_cos = (r + base -> w * d) / (1 + base -> w * base -> w); 
    msp -> m_cos = (r - base -> w * d) / (1 + base -> w * base -> w); 
}

void Conuix::get_base(Conuix::kinematic_t* jet, Conuix::kinematic_t* lep, Conuix::base_t* bs){
    //NOTE: omega = (beta_mu / beta_jet - cos(theta)) /sin(theta) 
    bs -> w  = ((lep -> beta / jet -> beta) - bs -> cos) / bs -> sin; 
    bs -> w2 = bs -> w * bs -> w; 

    //NOTE: Omega^2 = w^2 + 1 - b^2_mu
    bs -> o2 = bs -> w2 + 1 - lep -> beta * lep -> beta; 
    bs -> o  = std::sqrt(bs -> o2);

    bs -> rbl  = lep -> beta / jet -> beta; 
    bs -> beta = lep -> beta;
    bs -> mass = lep -> mass;
    bs -> E    = lep -> energy; 
    
    bs -> tpsi = bs -> w;
    bs -> cpsi = 1.0 / std::sqrt(1 + bs -> w * bs -> w); 
    bs -> spsi = bs -> w * bs -> cpsi; 
}


void Conuix::get_pencil(
        Conuix::kinematic_t* lep, Conuix::kinematic_t* nu, 
        Conuix::base_t* base, Conuix::pencil_t* pen
){
    // Z^2 = a Sx^2 + b Sx Sy + c Sy^2 + d Sx + e 
    pen -> a = (1 - base -> o2) / base -> o2;
    pen -> b = 2 * base -> w / base -> o2;
    pen -> c = (base -> w2 - base -> o2) / base -> o2; 
    pen -> d = 2 * lep -> energy * lep -> beta; 
    pen -> e = lep -> mass * lep -> mass - nu -> mass * nu -> mass; 
}

void Conuix::get_sx(Conuix::base_t* base, Conuix::Sx_t* sx){
    // Sx(tau) = Z [ (Omega / beta_mu) cos(psi) cosh(tau) - sin(psi) sinh(tau)] - m^2_mu / (E_mu beta_mu)
    sx -> a =  (base -> cpsi / base -> beta) * base -> o;
    sx -> b = - base -> spsi; 
    sx -> c = - base -> mass * base -> mass / (base -> E * base -> beta); 
}

void Conuix::get_sy(Conuix::base_t* base, Conuix::Sy_t* sy){
    // Sy(tau) = Z [(sin(psi) / beta_mu) Omega cosh(tau) + cos(psi) sinh(tau)] - tan(psi) E_mu / beta_mu
    sy -> a = (base -> spsi / base -> beta) * base -> o;
    sy -> b =  base -> cpsi; 
    sy -> c = - (base -> E / base -> beta) * base -> tpsi; 
}


void Conuix::get_hmatrix(Conuix::base_t* base, Conuix::rotation_t* rot, Conuix::H_matrix_t* H){
    // .......... Now perform a preliminary rotation .......... //
    // to get HMatrix -> i.e. the transformation matrix of the ellipse.

    // ----------- build static matrices ----------- //
    H -> HBX.at(0, 0) = 1            / base -> o; 
    H -> HBX.at(1, 0) = base -> tpsi / base -> o; 
    H -> HBX.at(2, 0) = 0;

    H -> HBX.at(0, 1) = 0; 
    H -> HBX.at(1, 1) = 0;
    H -> HBX.at(2, 1) = 1;

    H -> HBX.at(0, 2) = 0;
    H -> HBX.at(1, 2) = 0;
    H -> HBX.at(2, 2) = 0; 
    H -> HTX = rot -> R_T.dot(H -> HBX); 

    // .......... cosh matrix.............. //
    H -> HBC.at(0, 0) = 0; 
    H -> HBC.at(1, 0) = 0; 
    H -> HBC.at(2, 0) = 0;

    H -> HBC.at(0, 1) = 0; 
    H -> HBC.at(1, 1) = 0;
    H -> HBC.at(2, 1) = 0;

    H -> HBC.at(0, 2) = base -> beta * base -> cpsi / base -> o;
    H -> HBC.at(1, 2) = base -> beta * base -> spsi / base -> o;
    H -> HBC.at(2, 2) = 0; 
    H -> HTC = rot -> R_T.dot(H -> HBC);

    // .......... sinh matrix.............. //
    H -> HBS.at(0, 0) = 0; 
    H -> HBS.at(1, 0) = 0; 
    H -> HBS.at(2, 0) = 0;

    H -> HBS.at(0, 1) = 0; 
    H -> HBS.at(1, 1) = 0;
    H -> HBS.at(2, 1) = 0;

    H -> HBS.at(0, 2) =   base -> spsi;
    H -> HBS.at(1, 2) = - base -> cpsi;
    H -> HBS.at(2, 2) = 0; 
    H -> HTS = rot -> R_T.dot(H -> HBS); 
}


