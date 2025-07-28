#include "nusol.h"
#include <cmath>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static Mat3 rotation_matrix(int axis, double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    switch (axis) {
        case 0: return Mat3({ {1,0,0}, {0,c,-s}, {0,s,c} });
        case 1: return Mat3({ {c,0,s}, {0,1,0}, {-s,0,c} });
        case 2: return Mat3({ {c,-s,0}, {s,c,0}, {0,0,1} });
        default: return Mat3();
    }
}

NuSol::NuSol(const Vec4& b, const Vec4& lep, double mt, double mw): 
    b_jet(b), lepton(lep), mT(mt), mW(mw)
{
    this -> calculate_kinematics();
    this -> bx = new Boundary(this); 
}

std::vector<std::vector<double>> NuSol::get_boundary(){
    std::vector<std::vector<double>> bounds = {};
    this -> bx -> bounds(&bounds); 
    abort();  
    return bounds; 
}



void NuSol::calculate_kinematics() {
    double Eb     = this -> b_jet.e;
    double Em     = this -> lepton.e;
    double mb2    = this -> b_jet.mass2(); 
    double mm2    = this -> lepton.mass2();
    double Pm_val = this -> lepton.p();
    double Pb_val = this -> b_jet.p();
    this -> Bm = this -> lepton.beta();
    this -> Bb = this -> b_jet.beta();

    if (Eb <= 0 || Em <= 0 || Pb_val <= 0 || Pm_val <= 0) {
        Z2 = -1.0;
        return;
    }

    cos_th = b_jet.pvec().dot(lepton.pvec()) / (Pb_val * Pm_val);
    double sin_th_sq = 1.0 - cos_th * cos_th;
    if (sin_th_sq < 1e-12) {
        Z2 = -1.0;
        return;
    }

    sin_th = std::sqrt(sin_th_sq);
    double mT2 = this -> mT * this -> mT;
    double mW2 = this -> mW * this -> mW;
    x0p = -(mT2 - mW2 - mb2) / (2.0 * Eb);
    x0  = -(mW2 - mm2      ) / (2.0 * Em);

    Sx = (x0 * Bm - Pm_val * (1.0 - Bm * Bm)) / Bm;
    Sy = (x0p / Bb - cos_th * Sx) / sin_th;

    w = (Bm / Bb - cos_th) / sin_th;
    om2 = w * w + 1.0 - Bm * Bm;
    eps2 = mW2 * (1.0 - Bm * Bm);
    if (std::abs(om2) < 1e-12) {Z2 = -1.0; return;}

    x1 = Sx - (Sx + w * Sy) / om2;
    y1 = Sy - (Sx + w * Sy) * (w / om2);
    Z2 = x1 * x1 * om2 - (Sy - w * Sx) * (Sy - w * Sx) - (mW2 - x0 * x0 - eps2);
    if (std::isfinite(Z2) && Z2 >= 0){Z = std::sqrt(Z2);} 
    else {Z2 = -1.0;}
}

Mat3 NuSol::calculate_R_T() {
    Mat3 R_z = rotation_matrix(2, -lepton.phi());
    Mat3 R_y = rotation_matrix(1, 0.5 * M_PI - lepton.theta());
    Vec3 rotated_b = R_y * (R_z * b_jet.pvec());
    double alpha = -std::atan2(rotated_b.z, rotated_b.y);
    Mat3 R_x = rotation_matrix(0, alpha);
    return (R_x * R_y * R_z).transpose();
}

std::pair<bool, Mat3> NuSol::getH() {
    if (Z2 < 1e-9 || std::abs(om2) < 1e-9) { return {false, Mat3()}; }
    Mat3 base({ {Z / om2, 0, x1}, {w * Z / om2, 0, y1}, {0, Z, 0} });
    return {true, calculate_R_T() * base};
}

std::pair<bool, std::pair<Mat3, Mat3>> NuSol::getH_derivatives() {
    if (Z2 < 1e-9 || std::abs(om2) < 1e-9) { return {false, {}}; }

    double Eb = b_jet.e;
    double Em = lepton.e;
    double Bm2 = Bm * Bm;
    double two_mW = 2.0 * mW;
    double two_mT = 2.0 * mT;

    double common_inv_om2 = 1.0 / om2;
    double Sy_wSx = Sy - w * Sx;

    double dx0_dmW2 = -0.5 / Em;
    double dSx_dmW2 = dx0_dmW2;
    double dx0p_dmW2 = 0.5 / Eb;
    double dSy_dmW2 = (dx0p_dmW2 / Bb - cos_th * dSx_dmW2) / sin_th;
    double deps2_dmW2 = 1.0 - Bm2;

    double dx1_dmW2 = dSx_dmW2 - (dSx_dmW2 + w * dSy_dmW2) * common_inv_om2;
    double dZ2_dmW  = two_mW * (x1 * dx1_dmW2 * om2 - Sy_wSx * (dSy_dmW2 - w * dSx_dmW2) - (0.5 - x0 * dx0_dmW2 - 0.5 * deps2_dmW2));
    double dZ_dmW = (Z > 1e-9) ? 0.5 * dZ2_dmW / Z : 0.0;
    double dx1_dmW = dx1_dmW2 * two_mW;
    double dy1_dmW = (dSy_dmW2 - w * dSx_dmW2) * two_mW;

    double dx0p_dmT2 = -0.5 / Eb;
    double dSy_dmT2 = (dx0p_dmT2 / Bb) / sin_th;
    double dx1_dmT2 = -w * dSy_dmT2 * common_inv_om2;

    double dZ2_dmT = two_mT * (x1 * dx1_dmT2 * om2 - Sy_wSx * dSy_dmT2);
    double dZ_dmT = (Z > 1e-9) ? 0.5 * dZ2_dmT / Z : 0.0;
    double dx1_dmT = dx1_dmT2 * two_mT;
    double dy1_dmT = dSy_dmT2 * two_mT;
    Mat3 R_T = calculate_R_T();

    Mat3 dBase_dmT({ 
            {dZ_dmT * common_inv_om2    , 0     , dx1_dmT},
            {w * dZ_dmT * common_inv_om2, 0     , dy1_dmT},
            {0                          , dZ_dmT,       0}
    });
    Mat3 dBase_dmW({ 
            {dZ_dmW * common_inv_om2    , 0     , dx1_dmW},
            {w * dZ_dmW * common_inv_om2, 0     , dy1_dmW},
            {0                          , dZ_dmW, 0      } 
    });
    return {true, {R_T * dBase_dmT, R_T * dBase_dmW}};
}

std::pair<bool, Vec2> NuSol::getZ2Derivatives() const {
    if (Z2 < 0) { return {false, {}}; }
    double Eb = b_jet.e;
    double Em = lepton.e;
    double i_o2 = 1.0 / om2;
    double Sy_wSx = Sy - w * Sx;
    double dZ2_dmT = 2.0 * mT * mT * (x1 * w + Sy_wSx) / (Eb * this -> Bb * sin_th);

    double dx0_dmW   = -mW / Em;
    double dSy_dmW   = (mW / (Eb * this -> Bb) - cos_th * dx0_dmW) / sin_th;
    double dZ2_dmW   = 2.0 * mW * ( x1 * (dx0_dmW - (dx0_dmW + w * dSy_dmW) * i_o2) * om2 
            - Sy_wSx * (dSy_dmW - w * dx0_dmW) 
            - (-0.5 - x0 * dx0_dmW + this -> Bm * this -> Bm) );

    return {true, Vec2(dZ2_dmT, dZ2_dmW)};
}

Boundary::Boundary(const NuSol* nu):nu(nu){
    this -> analytical_boundary();  
}

Boundary::~Boundary(){}
double Boundary::Z2(double mW, double mT){
    double E_b = this -> nu -> b_jet.e; 
    double E_m = this -> nu -> lepton.e;

    double P_b = this -> nu -> b_jet.p(); 
    double P_m = this -> nu -> lepton.p(); 

    double B_b = this -> nu -> b_jet.beta(); 
    double B_m = this -> nu -> lepton.beta(); 

    double mm2 = this -> nu -> lepton.mass2();  
    double mb2 = this -> nu -> b_jet.mass2(); 
   
    double c = this -> nu -> b_jet.pvec().dot(this -> nu -> lepton.pvec()) / (P_b * P_m);
    double s = 1.0 - c * c;
    if (s < 0.0){return -1.0;}

    s = std::sqrt(s); 
    double w  = (B_m / B_b - c)/s;
    double o2 = w * w + 1.0 - B_m * B_m;
    
    double mT2 = mT * mT; 
    double mW2 = mW * mW; 

    double x0p  = -(mT2 - mW2 - mb2) / (2.0 * E_b);
    double x0   = -(mW2 - mm2) / (2.0 * E_m);
    double eps2 = mW2 * (1.0 - B_m * B_m);

    double Sx = x0 - P_m * (1 - B_m * B_m) / B_m;
    double Sy = (x0p / B_b - c * Sx) / s;
    
    double x1  = Sx - (Sx + w * Sy) / o2;
    double sqx = Sy - w * Sx;
    
    return x1 * x1 * o2 - sqx * sqx - (mW2 - x0 * x0 - eps2);
}

void Boundary::DZ2(double mW, double mT){
    double E_b     = this -> nu -> b_jet.e; 
    double E_mu    = this -> nu -> lepton.e; 
    double P_b     = this -> nu -> b_jet.p();
    double P_mu    = this -> nu -> lepton.p();
    double beta_b  = this -> nu -> b_jet.beta();
    double beta_mu = this -> nu -> lepton.beta();
    double c = this -> nu -> b_jet.pvec().dot(this -> nu -> lepton.pvec()) / (P_b * P_mu);
    double s = std::sqrt(std::max(0.0, 1.0 - c*c));
    if (std::abs(s) < 1e-9){return;}

    double w = (beta_mu / beta_b - c) / s;
    double o2 = w*w + 1 - beta_mu*beta_mu;
    if (std::abs(o2) < 1e-9){return;}
    
    double x0p = -(mT * mT - mW * mW - this -> nu -> b_jet.mass2()) / (2.0*E_b);
    double x0  = -(mW * mW - this -> nu -> lepton.mass2()) / (2.0*E_mu);
    double Sx = x0 - P_mu * (1 - beta_mu*beta_mu) / beta_mu;
    double Sy = (x0p/beta_b - c*Sx)/s;
    double x1 = Sx - (Sx + w*Sy)/o2;

    double dx0_dmW = -mW / E_mu;
    double dSy_dmW = (mW/P_b - c*dx0_dmW)/s;
    double dx1_dmW = dx0_dmW * (1 - 1/o2) - (w/o2)*dSy_dmW;
    double d_eps   = mW * (x0/E_mu + beta_mu*beta_mu);
    this -> jacobian[0] = 2.0 * ((x1 - Sx) * w + Sy) * mT/ (P_b * s);  // dmT
    this -> jacobian[1] = 2.0 * (x1*dx1_dmW*o2 - (Sy-w*Sx)*(dSy_dmW - w*dx0_dmW) - d_eps); //dmW    
}

void Boundary::D2Z2(){
    this -> hessian[0] = 0; this -> hessian[1] = 0; this -> hessian[2] = 0; 
    double E_b = this -> nu -> b_jet.e, P_b = this -> nu -> b_jet.p();
    double E_mu = this -> nu -> lepton.e;
    if (E_b == 0 || E_mu == 0){return;}
    double beta_b = this -> nu -> b_jet.beta(), beta_mu = this -> nu -> lepton.beta();
    double c = this -> nu -> b_jet.pvec().dot(this -> nu -> lepton.pvec()) / (P_b * this -> nu -> lepton.p());
    double s = std::sqrt(std::max(0.0, 1.0 - c * c));
    if (std::abs(s) < 1e-9){return;}
    double w = (beta_mu / beta_b - c) / s;
    double Om2 = w * w + 1.0 - beta_mu * beta_mu;
    if (std::abs(Om2) < 1e-9){return;}
    double A1 = -1.0 / (2.0 * E_mu);
    double B1 = -1.0 / (2.0 * s * E_b);
    double B2 = (1.0 / (2.0 * s * E_b)) - (c / s) * A1;
    
    double dx1_dw = (1.0 - 1.0 / Om2) * A1 - (w / Om2) * B2;

    this -> hessian[0] = -2.0 * B1 * B1 * (1.0 - w * w / Om2); // | D2 Z2 / (d2 mT2)
    this -> hessian[1] =  2.0 * B1 * (w * A1 / Om2 - (1.0 - w * w / Om2) * B2); // | D2 Z2 / dmW2 dmT2
//    this -> hessian[2] =  2.0 * (Om2 * dx1_dw * dx1_dw - std::pow(B2 - w * A1, 2)) + 1.0 / (2.0 * E_mu * E_mu); // | D2 Z2 / (d2 dmW2)
    this -> hessian[2] = 2.0 * Om2 * pow((1 - 1/Om2)*A1 - w * B2 / Om2, 2) - 2.0 * pow(B2 - w * A1, 2) + 1.0/(2*E_mu*E_mu);
}

void Boundary::analytical_boundary(){
    this -> DZ2(0, 0); this -> D2Z2(); 
    double A = this -> hessian[0];
    double B = this -> hessian[2];
    double C = 2.0 * this -> hessian[1];   
    double D = this -> jacobian[0]; 
    double E = this -> jacobian[1];
    double F = this -> Z2(0, 0); 

    // --- 3. Print the Final Analytical Expression ---
    std::cout << "Analytical Boundary Equation (Z² = 0):" << std::endl;
    std::cout << "Aλ_T² + Bλ_W² + Cλ_Tλ_W + Dλ_T + Eλ_W + F = 0" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "A (for λ_T²):      " << A << std::endl;
    std::cout << "B (for λ_W²):      " << B << std::endl;
    std::cout << "C (for λ_Tλ_W):    " << C << std::endl;
    std::cout << "D (for λ_T):       " << D << std::endl;
    std::cout << "E (for λ_W):       " << E << std::endl;
    std::cout << "F (Constant):    " << F << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    
    std::cout << "\n (" << A << ")λ_T²"; 
    std::cout << " + (" << B << ")λ_W²";
    std::cout << " + (" << C << ")λ_Tλ_W";
    std::cout << " + (" << D << ")λ_T";
    std::cout << " + (" << E << ")λ_W";
    std::cout << " + (" << F << ") = 0" << std::endl;
    this -> boundary[0][0] = A; this -> boundary[0][1] = B; this -> boundary[0][2] = D; 
    this -> boundary[1][0] = B; this -> boundary[1][1] = C; this -> boundary[1][2] = E;
    this -> boundary[2][0] = D; this -> boundary[2][1] = E; this -> boundary[2][2] = F; 
}

double Boundary::phi(){
    return 0.5 * std::atan2(this -> boundary[1][1], this -> boundary[0][0] - this -> boundary[0][1]);
}

std::pair<double, double> Boundary::Center(){
    // (2 * B * E - C * D)/(C^2 - 4 * A * B)
    double lambdaT_c = (2.0 * this -> boundary[0][1] * this -> boundary[2][1] - this -> boundary[1][1] * this -> boundary[2][0]); 
    lambdaT_c = lambdaT_c / (this -> boundary[1][1] * this -> boundary[1][1] - 4.0 * this -> boundary[0][0] * this -> boundary[0][1]); 
 
    // (2 * A * D - C * E)/(C^2 - 4 * A * B)
    double lambdaW_c = (2.0 * this -> boundary[0][0] * this -> boundary[2][0] - this -> boundary[1][1] * this -> boundary[1][2]); 
    lambdaW_c = lambdaW_c / (this -> boundary[1][1] * this -> boundary[1][1] - 4.0 * this -> boundary[0][0] * this -> boundary[0][1]); 
    return {lambdaT_c, lambdaW_c}; 
}

std::pair<double, double> Boundary::Axes(){
    std::pair<double, double> lambda_c = this -> Center(); 
    double A = this -> boundary[0][0];
    double B = this -> boundary[0][1]; 
    double C = this -> boundary[1][1];
    double D = this -> boundary[0][2]; 
    double E = this -> boundary[1][2];
    double F = this -> boundary[2][2]; 
    double l_Tc  = std::get<0>(lambda_c);
    double l_Wc  = std::get<1>(lambda_c); 

    double R2_tw = -2.0 * (A * l_Tc * l_Tc + B * l_Wc * l_Wc + C * l_Tc * l_Wc + D * l_Tc + E * l_Wc + F); 
    double scx   = pow((A - B)*(A - B) + C*C, 0.5); 
    double R2_t  = R2_tw / (A + B - scx); 
//    std::cout << R2_tw << " " << A + B - scx << std::endl;
    double R2_w  = R2_tw / (A + B + scx); 
    return {R2_t, R2_w}; 
}

void Boundary::bounds(std::vector<std::vector<double>>* inpt, int res){
    std::vector<double> lambdaT = {}; 
    std::vector<double> lambdaW = {}; 
    std::pair<double, double> tw_c = this -> Center(); 
    std::pair<double, double> tw_a = this -> Axes(); 
    double lambda_ct = std::get<0>(tw_c); 
    double lambda_cw = std::get<1>(tw_c); 
    double axis_t = std::sqrt(std::get<0>(tw_a));
    double axis_w = std::sqrt(std::get<1>(tw_a)); 
    double ro_phi = this -> phi(); 
    double c = std::cos(ro_phi);
    double s = std::sin(ro_phi); 
    double step = (1.0 / double(res))*2*M_PI; 
    std::cout << lambda_ct << " " << lambda_cw << " " << axis_t << " " << axis_w << " " << ro_phi << " " << c << std::endl;
    for (size_t x(0); x < res; ++x){
        double c_ = std::cos(double(x)*step); 
        double s_ = std::sin(double(x)*step); 

        double t = lambda_ct + axis_t * c * c_ - axis_w * s * s_; 
        double w = lambda_cw + axis_t * s * c_ + axis_w * c * s_;
        std::cout << t << " " << w << std::endl; 
    }
}









