#include <templates/particle_template.h>
#include <conuix/struct.h> 

double cos_theta(particle_template* jet, particle_template* lep){
    double d12 = 0;
    d12 += double(jet -> px) * double(lep -> px); 
    d12 += double(jet -> py) * double(lep -> py); 
    d12 += double(jet -> pz) * double(lep -> pz); 
    return d12 / (double(jet -> P) * double(lep -> P)); 
}


atomics_t::atomics_t(particle_template* jet, particle_template* lep, double m_nu){
    // ------------ get all the kinematics --------- //
    this -> beta_l = lep -> beta; 
    this -> mass_l = lep -> mass; 
    this -> e_l    = lep -> e; 

    this -> beta_j = jet -> beta; 
    this -> mass_j = jet -> mass;
    this -> e_j    = jet -> e; 

    // ----------- define base quantities --------- //
    this -> cos = cos_theta(jet, lep);
    this -> sin = std::pow(1 - this -> cos * this -> cos, 0.5); 
    
    //NOTE: omega = (beta_mu / beta_jet - cos(theta)) /sin(theta) 
    this -> w = ( (this -> beta_l / this -> beta_j) - this -> cos) / this -> sin; 

    //NOTE: Omega = sqrt(w^2 + 1 - b^2_mu) -> sqrt( w^2 + (m/e)^2 )
    this -> o = std::pow(std::pow(this -> w, 2) + std::pow(this -> mass_l / this -> e_l, 2), 0.5); 


    // ----------- mapping from psi to theta -------- //
    double r = this -> beta_l / this -> beta_j; 
    double d = std::pow(1 + this -> w * this -> w - r * r, 0.5); 
    this -> p_psi_sin = (r * this -> w + d) / (1 + this -> w * this -> w); 
    this -> m_psi_sin = (r * this -> w - d) / (1 + this -> w * this -> w); 

    this -> p_psi_cos = (r + this -> w * d) / (1 + this -> w * this -> w); 
    this -> m_psi_cos = (r - this -> w * d) / (1 + this -> w * this -> w); 

    // ---------- Pencil Function Surface Polynomial --------- //
    this -> Z2.a = (1 - this -> o * this -> o)/(this -> o * this -> o); 
    this -> Z2.b = std::pow(this -> mass_l / (this -> e_l * this -> o), 2); 
    this -> Z2.c = 2 * this -> w / (this -> o * this -> o); 
    this -> Z2.d = 2 * this -> e_l * this -> beta_l; 
    this -> Z2.e = this -> mass_l * this -> mass_l - m_nu * m_nu; 
    this -> Z2.len = 5; 

    // ---------- Hyperbolic rotation --------------- //
    this -> cpsi = std::pow(1.0 / (1 + this -> w * this -> w), 0.5); 
    this -> spsi = this -> w  / std::pow(1 + this -> w * this -> w, 0.5); 
    this -> tpsi = this -> w; 

    // ---------- Shift parameters -------------- //
    this -> Sx.a =  this -> cpsi * this -> o / this -> beta_l;
    this -> Sx.b = -this -> spsi;  
    this -> Sx.c = - (this -> mass_l * this -> mass_l) / (this -> e_l * this -> beta_l); 
    this -> Sx.len = 3; 

    this -> Sy.a =  this -> spsi * this -> o / this -> beta_l;
    this -> Sy.b =  this -> cpsi;  
    this -> Sy.c = -this -> w * this -> e_l / (this -> beta_l); 
    this -> Sy.len = 3; 
 
    // ----------- Definition of the rotation matrix defined in neutrino paper --------- // 
    this -> phi_mu   = std::atan2(lep -> py, lep -> px); 
    this -> theta_mu = std::acos(lep -> pz / (this -> e_l * this -> beta_l)); 
    this -> vec_jet  = matrix_t(3, 1);
    this -> vec_jet.at(0, 0) = jet -> px;
    this -> vec_jet.at(1, 0) = jet -> py;
    this -> vec_jet.at(2, 0) = jet -> pz; 

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

    // ----------- build static matrices ----------- //
    this -> HBX.at(0, 0) = 1            / this -> o; 
    this -> HBX.at(1, 0) = this -> tpsi / this -> o; 
    this -> HBX.at(2, 0) = 0;

    this -> HBX.at(0, 1) = 0; 
    this -> HBX.at(1, 1) = 0;
    this -> HBX.at(2, 1) = 1;

    this -> HBX.at(0, 2) = 0;
    this -> HBX.at(1, 2) = 0;
    this -> HBX.at(2, 2) = 0; 

    // .......... cosh matrix.............. //
    this -> HBC.at(0, 0) = 0; 
    this -> HBC.at(1, 0) = 0; 
    this -> HBC.at(2, 0) = 0;

    this -> HBC.at(0, 1) = 0; 
    this -> HBC.at(1, 1) = 0;
    this -> HBC.at(2, 1) = 0;

    this -> HBC.at(0, 2) = - this -> beta_l * this -> cpsi / this -> o;
    this -> HBC.at(1, 2) = - this -> beta_l * this -> spsi / this -> o;
    this -> HBC.at(2, 2) = 0; 

    // .......... sinh matrix.............. //
    this -> HBS.at(0, 0) = 0; 
    this -> HBS.at(1, 0) = 0; 
    this -> HBS.at(2, 0) = 0;

    this -> HBS.at(0, 1) = 0; 
    this -> HBS.at(1, 1) = 0;
    this -> HBS.at(2, 1) = 0;

    this -> HBS.at(0, 2) = - this -> spsi;
    this -> HBS.at(1, 2) =   this -> cpsi;
    this -> HBS.at(2, 2) = 0; 

    // .......... Now perform a preliminary rotation .......... //
    // to get HMatrix -> i.e. the transformation matrix of the ellipse.
    this -> HMX = this -> R_T.dot(this -> HBX); 
    this -> HMC = this -> R_T.dot(this -> HMC);
    this -> HMS = this -> R_T.dot(this -> HMS); 


    // ............ other stuff ............... //
    // -- hyperbolic function 
    this -> gxx_a = this -> beta_l * this -> spsi; 
    this -> gxx_b = this -> o      * this -> cpsi; 

    // dP/dtau.
    this -> gtx_a =  this -> cpsi * this -> o;
    this -> gtx_b = -this -> spsi * this -> beta_l; 

    // -- characteristic polynomials
    // P.
    this -> p_a = - 1;
    this -> p_b =   1.0 / this -> o;
    this -> p_c = - 1.0 / this -> o; 
    this -> p_d = - 1.0 / (this -> o * this -> cpsi); 

    // dP/dL.
    this -> dpdl_a = -3; 
    this -> dpdl_b =  2 / this -> o;
    this -> dpdl_c = -1 / this -> o; 

    // dP/dZ.
    this -> dpdz_a =  1 / this -> o; 
    this -> dpdz_b = -2 / this -> o;
    this -> dpdz_c = -3 / (this -> o * this -> cpsi); 

    // dP/dt.
    this -> dpdt_a = -1 / this -> o; 
    this -> dpdt_b =  1 / (this -> cpsi); 
    
    // ............ WARNING ENTER AT YOUR OWN RISK .............. //
    // Mobius Transformation: See header
    this -> M_pm = this -> o      * this -> cpsi;
    this -> M_pp = this -> beta_l * this -> spsi; 
    this -> M_km = this -> o      * this -> spsi;
    this -> M_kp = this -> beta_l * this -> cpsi; 
   
    // ---- cosh(tau) / kappa(tau) + beta_mu * cos(psi) * M(tau)^2 = 0 --- // 
    this -> M_r = this -> cpsi * this -> beta_l; 

    // ..... quartic coefficients
    // a = 1
    // b = - 2 Omega / (beta_mu * tan(psi))
    // c = (Omega / ( beta_mu * tan(psi)))^2 - 1
    // d = 2 * Omega / (beta_mu * tan(psi))
    // e = ( [ (1 + tan(psi)^2 ] / [ (beta_mu tan(psi))^2 ] )^2 - [ Omega / (beta_mu tan(psi)) ]^2
    long double rM = this -> o / (this -> beta_l * this -> tpsi);  
    this -> M_qrt.a = 1;
    this -> M_qrt.b = - 2 * rM; 
    this -> M_qrt.c = rM * rM - 1; 
    this -> M_qrt.d = 2 * rM;
    this -> M_qrt.e = ( (1 + this -> tpsi * this -> tpsi) / std::pow(this -> beta_mu * this -> tpsi, 2) );
    this -> M_qrt.e = std::pow(this -> M_qrt.e, 2) - rM * rM;


}



