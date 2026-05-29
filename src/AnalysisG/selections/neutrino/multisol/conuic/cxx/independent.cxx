#include <conuic/atomics.h>
#include <conuic/conuic.h>

long double conuic::mW2(long double sx, long double m_nu){
    shared_t* sh = this -> shr; 
    return pw(m_nu) - pw(sh -> m_mu) - 2 * sh -> p_mu * sx; 
}

long double conuic::mT2(long double sx, long double sy, long double m_nu){
    shared_t* sh = this -> shr; 
    long double o = pw(m_nu) + pw(sh -> m_bq) - pw(sh -> m_mu);
    o -= 2 * (sh -> p_mu + sh -> p_bq * sh -> cos) * sx; 
    o -= 2 * sh -> p_bq * sh -> sin * sy; 
    return o; 
}

long double conuic::Sx(long double mW, long double m_nu){
    shared_t* sh = this -> shr; 
    return - (pw(mW) - pw(m_nu) + pw(sh -> m_mu)) / (2 * sh -> p_mu); 
}

long double conuic::Sy(long double mt, long double mW, long double m_nu){
    shared_t* sh = this -> shr; 
    long double o = pw(mt) - pw(sh -> m_bq) - pw(m_nu) + pw(sh -> m_mu); 
    o += 2 * (sh -> p_mu + sh -> p_bq * sh -> cos) * this -> Sx(mW, m_nu); 
    return - o / (2 * sh -> p_bq * sh -> sin); 
}

long double conuic::Z2(long double sx, long double sy, long double m_nu, int sign){
    base_t* bx = this -> branching(sign);
    long double o = bx -> A * sx * sx; 
    o += bx -> B * sx * sy;
    o += bx -> C * sy * sy;
    o += bx -> D * sx;
    o += bx -> E - m_nu * m_nu; 
    return o; 
}

long double conuic::Z2lxly(long double lx, long double ly, long double m_nu, long double sign){
    base_t* bx = this -> branching(sign); 
    pk1l_t* kx = this -> pl1; 
    long double eta_ = kx -> eta; 
    long double seth = std::sinh(eta_);
    long double ceth = std::cosh(eta_);
    long double cet2 = std::cosh(2.0L * eta_); 

    long double O    = bx -> O; 
    long double wg   = bx -> w / kx -> gmu; 

    long double A_ = ((1.0L - O * O) * pw(seth) + 2 * wg * seth * ceth - pw(ceth)) / pw(kx -> GP * O * cet2);
    long double B_ = (seth * ceth * (2.0L - O*O) + wg)/(kx -> GP * kx -> GM * pw(O * cet2));  
    long double C_ = ((1.0L - O * O) * pw(ceth) - 2 * wg * seth * ceth - pw(seth)) / pw(kx -> GM * O * cet2); 
    
    long double z2_ = A_ * lx * lx + 2 * B_ * lx * ly + C_ * ly * ly; 
    z2_ += (bx -> D * seth / (kx -> GP * cet2)) * lx; 
    z2_ += (bx -> D * ceth / (kx -> GM * cet2)) * ly; 
    z2_ += (bx -> E - m_nu * m_nu); 
    return z2_;  
}

long double conuic::g2(long double sx, long double sy){
    pk1l_t* bx = this -> pl1; 
    return - bx -> GP * bx -> GM * bx -> lx(sx, sy) * bx -> ly(sx, sy); 
}

long double conuic::G2(long double sx, long double sy){
    pk1l_t* bx = this -> pl1; 
    long double f = - 1 / (std::sinh(bx -> eta) * std::cosh(bx -> eta)); 
    return f * bx -> Lx(sx, sy) * bx -> Ly(sx, sy); 
}

